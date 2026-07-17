#include <torch/extension.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::mutex pipeline_mutex;
struct Pipeline {
  id<MTLComputePipelineState> state;
  id<MTLArgumentEncoder> argument_encoder = nil;
  enum class ArgumentType : uint8_t { Buffer, Float32, Int32 };
  std::vector<ArgumentType> argument_types;
  std::vector<MTLResourceUsage> argument_usage;
};
std::vector<std::shared_ptr<Pipeline>> pipelines;

struct ArgumentBinding {
  id<MTLBuffer> encoded;
  std::vector<std::pair<id<MTLBuffer>, MTLResourceUsage>> resources;
  std::vector<id<MTLBuffer>> owned_scalar_buffers;
  std::vector<torch::Tensor> retained_tensors;
  ~ArgumentBinding() {
    [encoded release];
    for (id<MTLBuffer> buffer : owned_scalar_buffers) [buffer release];
  }
};
std::mutex binding_mutex;
std::vector<std::shared_ptr<ArgumentBinding>> bindings;

struct ICBGraph {
  id<MTLIndirectCommandBuffer> commands;
  std::vector<std::pair<id<MTLBuffer>, MTLResourceUsage>> resources;
  std::vector<std::shared_ptr<ArgumentBinding>> retained_bindings;
  NSUInteger command_count;
  ~ICBGraph() {
    [commands release];
  }
};
std::mutex graph_mutex;
std::vector<std::shared_ptr<ICBGraph>> graphs;

std::shared_ptr<Pipeline> get_pipeline(int64_t pipeline_id);

MTLLanguageVersion latest_stable_msl_version() {
  // Keep the binary buildable with older SDKs while selecting the newest
  // stable language revision supported by both the build SDK and host OS.
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (@available(macOS 26.0, *)) return MTLLanguageVersion4_0;
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  if (@available(macOS 15.0, *)) return MTLLanguageVersion3_2;
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
  if (@available(macOS 14.0, *)) return MTLLanguageVersion3_1;
#endif
  TORCH_CHECK(@available(macOS 13.0, *),
              "Hydroforge Metal kernels require macOS 13 or newer (MSL 3.0)");
  return MTLLanguageVersion3_0;
}

int64_t compile_pipeline(
    const std::string& source,
    const std::string& kernel_name,
    const std::vector<std::pair<uint32_t, bool>>& bool_constants,
    const std::vector<std::string>& argument_types,
    const std::vector<std::string>& argument_access) {
  @autoreleasepool {
    id<MTLDevice> device = at::mps::getCurrentMPSStream()->device();
    NSString* metal_source = [NSString stringWithUTF8String:source.c_str()];
    NSError* error = nil;
    MTLCompileOptions* compile_options = [[MTLCompileOptions alloc] init];
    // Avoid the unreliable implicit default in a Torch JIT extension.  The
    // kernels require at least MSL 3.0 for atomic_float.
    compile_options.languageVersion = latest_stable_msl_version();
    id<MTLLibrary> library = [device newLibraryWithSource:metal_source
                                                  options:compile_options
                                                    error:&error];
    [compile_options release];
    TORCH_CHECK(library != nil, "Metal library compilation failed: ",
                error ? error.localizedDescription.UTF8String : "unknown error");

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    for (const auto& [index, value] : bool_constants) {
      bool copy = value;
      [constants setConstantValue:&copy type:MTLDataTypeBool atIndex:index];
    }
    NSError* function_error = nil;
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]
             constantValues:constants
                      error:&function_error];
    TORCH_CHECK(function != nil, "Metal function specialization failed: ",
                function_error
                    ? function_error.localizedDescription.UTF8String
                    : "unknown error");

    MTLComputePipelineDescriptor* descriptor =
        [[MTLComputePipelineDescriptor alloc] init];
    descriptor.computeFunction = function;
    descriptor.supportIndirectCommandBuffers = YES;
    NSError* pipeline_error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithDescriptor:descriptor
                                               options:MTLPipelineOptionNone
                                            reflection:nil
                                                 error:&pipeline_error];
    [descriptor release];
    id<MTLArgumentEncoder> argument_encoder =
        [function newArgumentEncoderWithBufferIndex:0];
    TORCH_CHECK(argument_encoder != nil,
                "Failed to create Metal argument encoder for ", kernel_name);
    [function release];
    [constants release];
    [library release];
    TORCH_CHECK(pipeline != nil, "Metal pipeline creation failed: ",
                pipeline_error
                    ? pipeline_error.localizedDescription.UTF8String
                    : "unknown error");
    std::lock_guard<std::mutex> guard(pipeline_mutex);
    std::vector<Pipeline::ArgumentType> encoded_types;
    encoded_types.reserve(argument_types.size());
    for (const auto& kind : argument_types) {
      if (kind == "buffer") encoded_types.push_back(Pipeline::ArgumentType::Buffer);
      else if (kind == "float32") encoded_types.push_back(Pipeline::ArgumentType::Float32);
      else if (kind == "int32") encoded_types.push_back(Pipeline::ArgumentType::Int32);
      else TORCH_CHECK(false, "Unsupported Metal argument type: ", kind);
    }
    auto item = std::make_shared<Pipeline>();
    item->state = pipeline;
    item->argument_encoder = argument_encoder;
    item->argument_types = std::move(encoded_types);
    TORCH_CHECK(argument_access.size() == argument_types.size(),
                "Metal argument access/type count mismatch");
    for (const auto& access : argument_access) {
      if (access == "read") item->argument_usage.push_back(MTLResourceUsageRead);
      else if (access == "write") item->argument_usage.push_back(MTLResourceUsageWrite);
      else if (access == "read_write") item->argument_usage.push_back(
          MTLResourceUsageRead | MTLResourceUsageWrite);
      else if (access == "none") item->argument_usage.push_back(MTLResourceUsageRead);
      else TORCH_CHECK(false, "Unsupported Metal resource access: ", access);
    }
    pipelines.push_back(std::move(item));
    return static_cast<int64_t>(pipelines.size() - 1);
  }
}

std::shared_ptr<ArgumentBinding> get_binding(int64_t binding_id) {
  std::lock_guard<std::mutex> guard(binding_mutex);
  TORCH_CHECK(binding_id >= 0 &&
                  static_cast<size_t>(binding_id) < bindings.size(),
              "Invalid Metal argument binding id: ", binding_id);
  auto binding = bindings[static_cast<size_t>(binding_id)];
  TORCH_CHECK(binding != nullptr,
              "Metal argument binding has been released: ", binding_id);
  return binding;
}

void release_argument_binding(int64_t binding_id) {
  std::shared_ptr<ArgumentBinding> binding;
  {
    std::lock_guard<std::mutex> guard(binding_mutex);
    if (binding_id < 0 || static_cast<size_t>(binding_id) >= bindings.size()) return;
    binding = std::move(bindings[static_cast<size_t>(binding_id)]);
  }
  binding.reset();
}

int64_t create_argument_binding(
    int64_t pipeline_id, const pybind11::list& arguments) {
  auto pipeline = get_pipeline(pipeline_id);
  TORCH_CHECK(arguments.size() == pipeline->argument_types.size(),
              "Metal argument/type count mismatch");
  id<MTLDevice> device = at::mps::getCurrentMPSStream()->device();
  auto binding = std::make_shared<ArgumentBinding>();
  binding->encoded = [device
      newBufferWithLength:pipeline->argument_encoder.encodedLength
                  options:MTLResourceStorageModeShared];
  [pipeline->argument_encoder setArgumentBuffer:binding->encoded offset:0];
  binding->resources.push_back({binding->encoded, MTLResourceUsageRead});
  for (pybind11::ssize_t i = 0; i < arguments.size(); ++i) {
    const auto kind = pipeline->argument_types[static_cast<size_t>(i)];
    pybind11::handle value = arguments[i];
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
    if (kind == Pipeline::ArgumentType::Buffer) {
      if (value.is_none()) {
        buffer = nil;
      } else {
        torch::Tensor tensor = pybind11::cast<torch::Tensor>(value);
        TORCH_CHECK(tensor.device().is_mps(),
                    "Metal argument buffers require MPS tensors");
        buffer = at::native::mps::getMTLBufferStorage(tensor);
        offset = tensor.storage_offset() * tensor.element_size();
        binding->retained_tensors.push_back(tensor);
      }
    } else if (kind == Pipeline::ArgumentType::Float32) {
      float scalar = pybind11::cast<float>(value);
      buffer = [device newBufferWithBytes:&scalar length:sizeof(scalar)
                                  options:MTLResourceStorageModeShared];
      binding->owned_scalar_buffers.push_back(buffer);
    } else {
      int32_t scalar = pybind11::cast<int32_t>(value);
      buffer = [device newBufferWithBytes:&scalar length:sizeof(scalar)
                                  options:MTLResourceStorageModeShared];
      binding->owned_scalar_buffers.push_back(buffer);
    }
    [pipeline->argument_encoder setBuffer:buffer offset:offset atIndex:i];
    if (buffer != nil) {
      binding->resources.push_back({
          buffer, pipeline->argument_usage[static_cast<size_t>(i)]});
    }
  }
  std::lock_guard<std::mutex> guard(binding_mutex);
  bindings.push_back(binding);
  return static_cast<int64_t>(bindings.size() - 1);
}

std::shared_ptr<Pipeline> get_pipeline(int64_t pipeline_id) {
  std::lock_guard<std::mutex> guard(pipeline_mutex);
  TORCH_CHECK(pipeline_id >= 0 &&
                  static_cast<size_t>(pipeline_id) < pipelines.size(),
              "Invalid Metal pipeline id: ", pipeline_id);
  return pipelines[static_cast<size_t>(pipeline_id)];
}

void dispatch(
    int64_t pipeline_id,
    int64_t binding_id,
    uint64_t threads,
    uint64_t requested_group_size) {
  auto pipeline = get_pipeline(pipeline_id);
  auto binding = get_binding(binding_id);
  if (threads == 0) return;

  auto* stream = at::mps::getCurrentMPSStream();
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline->state];
    [encoder setBuffer:binding->encoded offset:0 atIndex:0];
    for (const auto& [resource, usage] : binding->resources) {
      [encoder useResource:resource usage:usage];
    }
    NSUInteger width = std::min<NSUInteger>(
        requested_group_size, pipeline->state.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  });
}

void dispatch_sequence(
    const std::vector<int64_t>& pipeline_ids,
    const std::vector<int64_t>& binding_ids,
    const std::vector<uint64_t>& threads,
    const std::vector<uint64_t>& group_sizes,
    const std::vector<bool>& barriers) {
  const size_t count = pipeline_ids.size();
  TORCH_CHECK(binding_ids.size() == count && threads.size() == count &&
                  group_sizes.size() == count && barriers.size() == count,
              "Metal sequence arrays must have equal length");
  auto* stream = at::mps::getCurrentMPSStream();
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    for (size_t i = 0; i < count; ++i) {
      if (threads[i] == 0) continue;
      auto pipeline = get_pipeline(pipeline_ids[i]);
      auto binding = get_binding(binding_ids[i]);
      [encoder setComputePipelineState:pipeline->state];
      [encoder setBuffer:binding->encoded offset:0 atIndex:0];
      for (const auto& [resource, usage] : binding->resources) {
        [encoder useResource:resource usage:usage];
      }
      NSUInteger width = std::min<NSUInteger>(
          group_sizes[i], pipeline->state.maxTotalThreadsPerThreadgroup);
      [encoder dispatchThreads:MTLSizeMake(threads[i], 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      if (barriers[i]) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
    }
  });
}

int64_t create_icb(
    const std::vector<int64_t>& pipeline_ids,
    const std::vector<int64_t>& binding_ids,
    const std::vector<uint64_t>& threads,
    const std::vector<uint64_t>& group_sizes,
    const std::vector<bool>& barriers) {
  const size_t count = pipeline_ids.size();
  TORCH_CHECK(count > 0, "ICB requires at least one command");
  TORCH_CHECK(binding_ids.size() == count && threads.size() == count &&
                  group_sizes.size() == count && barriers.size() == count,
              "Metal ICB arrays must have equal length");
  auto* stream = at::mps::getCurrentMPSStream();
  id<MTLDevice> device = stream->device();
  NSUInteger max_bindings = 0;
  for (int64_t id : pipeline_ids) {
    max_bindings = std::max<NSUInteger>(max_bindings, 1);
  }

  MTLIndirectCommandBufferDescriptor* descriptor =
      [[MTLIndirectCommandBufferDescriptor alloc] init];
  descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatchThreads;
  descriptor.inheritPipelineState = NO;
  descriptor.inheritBuffers = NO;
  descriptor.maxKernelBufferBindCount = max_bindings;
  id<MTLIndirectCommandBuffer> commands =
      [device newIndirectCommandBufferWithDescriptor:descriptor
                                     maxCommandCount:count
                                             options:MTLResourceStorageModePrivate];
  [descriptor release];
  TORCH_CHECK(commands != nil, "Failed to allocate Metal indirect command buffer");

  auto graph = std::make_shared<ICBGraph>();
  graph->commands = commands;
  graph->command_count = count;
  std::unordered_set<void*> seen_resources;
  auto add_resource = [&](id<MTLBuffer> buffer, MTLResourceUsage usage) {
    void* key = (__bridge void*)buffer;
    if (seen_resources.insert(key).second) {
      graph->resources.push_back({buffer, usage});
    } else {
      for (auto& [existing, existing_usage] : graph->resources) {
        if (existing == buffer) {
          existing_usage = existing_usage | usage;
          break;
        }
      }
    }
  };

  for (size_t command_index = 0; command_index < count; ++command_index) {
    auto pipeline = get_pipeline(pipeline_ids[command_index]);
    auto binding = get_binding(binding_ids[command_index]);
    id<MTLIndirectComputeCommand> command =
        [commands indirectComputeCommandAtIndex:command_index];
    [command setComputePipelineState:pipeline->state];
    [command setKernelBuffer:binding->encoded offset:0 atIndex:0];
    graph->retained_bindings.push_back(binding);
    for (const auto& [resource, usage] : binding->resources) {
      add_resource(resource, usage);
    }
    NSUInteger width = std::min<NSUInteger>(
        group_sizes[command_index], pipeline->state.maxTotalThreadsPerThreadgroup);
    TORCH_CHECK(threads[command_index] > 0 && width > 0,
                "ICB dispatch dimensions must be positive");
    [command concurrentDispatchThreads:MTLSizeMake(threads[command_index], 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    if (barriers[command_index]) [command setBarrier];
  }
  std::lock_guard<std::mutex> guard(graph_mutex);
  graphs.push_back(graph);
  return static_cast<int64_t>(graphs.size() - 1);
}

void replay_icb(int64_t graph_id, uint64_t replays) {
  TORCH_CHECK(replays > 0, "ICB replay count must be positive");
  std::shared_ptr<ICBGraph> graph;
  {
    std::lock_guard<std::mutex> guard(graph_mutex);
    TORCH_CHECK(graph_id >= 0 && static_cast<size_t>(graph_id) < graphs.size(),
                "Invalid Metal ICB graph id: ", graph_id);
    graph = graphs[static_cast<size_t>(graph_id)];
    TORCH_CHECK(graph != nullptr, "Metal ICB graph has been released: ", graph_id);
  }
  auto* stream = at::mps::getCurrentMPSStream();
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    for (const auto& [resource, usage] : graph->resources) {
      [encoder useResource:resource usage:usage];
    }
    for (uint64_t replay = 0; replay < replays; ++replay) {
      [encoder executeCommandsInBuffer:graph->commands
                              withRange:NSMakeRange(0, graph->command_count)];
    }
  });
}

void release_icb(int64_t graph_id) {
  std::shared_ptr<ICBGraph> graph;
  {
    std::lock_guard<std::mutex> guard(graph_mutex);
    if (graph_id < 0 || static_cast<size_t>(graph_id) >= graphs.size()) return;
    graph = std::move(graphs[static_cast<size_t>(graph_id)]);
  }
  // Destroy buffers outside the registry lock. Releasing the same id is safe.
  graph.reset();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("compile_pipeline", &compile_pipeline);
  module.def("create_argument_binding", &create_argument_binding);
  module.def("release_argument_binding", &release_argument_binding);
  module.def("dispatch", &dispatch);
  module.def("dispatch_sequence", &dispatch_sequence);
  module.def("create_icb", &create_icb);
  module.def("replay_icb", &replay_icb, pybind11::arg("graph_id"),
             pybind11::arg("replays") = 1);
  module.def("release_icb", &release_icb);
}
