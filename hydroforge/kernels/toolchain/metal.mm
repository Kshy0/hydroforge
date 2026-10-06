#include <torch/extension.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>

#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace {

// Libraries and pipelines are process-lifetime compilation caches; the
// Python side keys them by source text and specialization.
std::mutex library_mutex;
std::vector<id<MTLLibrary>> libraries;

std::mutex pipeline_mutex;
struct Pipeline {
  id<MTLComputePipelineState> state = nil;
  id<MTLArgumentEncoder> argument_encoder = nil;
  enum class ArgumentType : uint8_t {
    Buffer, Float32, Int32, UInt32, Int64, Bool
  };
  std::vector<ArgumentType> argument_types;
  std::vector<MTLResourceUsage> argument_usage;
  ~Pipeline() {
    [state release];
    [argument_encoder release];
  }
};
std::vector<std::shared_ptr<Pipeline>> pipelines;

using at::mps::MPSStream;

// Metal retains neither argument-buffer contents nor resources declared with
// useResource:.  A released binding or ICB therefore stays alive until every
// stream that encoded it has completed its current command buffer.
struct EncodedOn {
  std::vector<MPSStream*> streams;
  void note(MPSStream* stream) {
    for (MPSStream* seen : streams) {
      if (seen == stream) return;
    }
    streams.push_back(stream);
  }
};

// Intentionally leaked: a completion handler may still run at process exit.
std::mutex* const retired_mutex = new std::mutex();
auto* const retired_resources = new std::vector<std::shared_ptr<void>>();

// Completion handlers only hand ownership back; Objective-C and Torch
// releases run here, on a host thread calling into this extension: every
// create, release, dispatch and replay drains what has completed.
void drain_retired_resources() {
  std::vector<std::shared_ptr<void>> completed;
  {
    std::lock_guard<std::mutex> guard(*retired_mutex);
    completed.swap(*retired_resources);
  }
}

void release_after_completion(std::shared_ptr<void> resource,
                              const EncodedOn& encoded_on) {
  for (MPSStream* stream : encoded_on.streams) {
    auto* holder = new std::shared_ptr<void>(resource);
    try {
      stream->addCompletedHandler(^(id<MTLCommandBuffer> completed) {
        (void)completed;
        std::lock_guard<std::mutex> guard(*retired_mutex);
        retired_resources->push_back(std::move(*holder));
        delete holder;
      });
    } catch (...) {
      delete holder;
      throw;
    }
  }
}

struct ArgumentBinding {
  int64_t pipeline_id;
  id<MTLBuffer> encoded = nil;
  std::vector<std::pair<id<MTLBuffer>, MTLResourceUsage>> resources;
  std::vector<id<MTLBuffer>> owned_scalar_buffers;
  std::vector<torch::Tensor> retained_tensors;
  EncodedOn encoded_on;
  ~ArgumentBinding() {
    [encoded release];
    for (id<MTLBuffer> buffer : owned_scalar_buffers) [buffer release];
  }
};
std::mutex binding_mutex;
std::unordered_map<int64_t, std::shared_ptr<ArgumentBinding>> bindings;
int64_t next_binding_id = 0;

struct ICBGraph {
  id<MTLIndirectCommandBuffer> commands;
  std::vector<std::pair<id<MTLBuffer>, MTLResourceUsage>> resources;
  std::vector<std::shared_ptr<ArgumentBinding>> retained_bindings;
  NSUInteger command_count;
  EncodedOn encoded_on;
  ~ICBGraph() {
    [commands release];
  }
};
std::mutex graph_mutex;
std::unordered_map<int64_t, std::shared_ptr<ICBGraph>> graphs;
int64_t next_graph_id = 0;

std::shared_ptr<Pipeline> get_pipeline(int64_t pipeline_id);

constexpr uint64_t kMaxMetalGridExtent =
    std::numeric_limits<uint32_t>::max();

void validate_grid_extent(uint64_t threads) {
  TORCH_CHECK(threads <= kMaxMetalGridExtent,
              "Metal launch extent exceeds the uint32 grid range: ", threads);
}

NSUInteger validate_group_size(const Pipeline& pipeline, uint64_t requested) {
  TORCH_CHECK(
      requested > 0 && requested <= pipeline.state.maxTotalThreadsPerThreadgroup,
      "Requested Metal threadgroup width ", requested,
      " exceeds pipeline limit ", pipeline.state.maxTotalThreadsPerThreadgroup);
  return static_cast<NSUInteger>(requested);
}

// Scalar arguments of one binding share one buffer, one slot each. A slot
// keeps every constant-address-space offset 256-byte aligned, as macOS
// requires of constant buffer offsets.
constexpr NSUInteger kScalarSlot = 256;

template <typename Scalar>
void write_scalar(id<MTLBuffer> buffer, NSUInteger offset, pybind11::handle value) {
  Scalar scalar = pybind11::cast<Scalar>(value);
  std::memcpy(static_cast<char*>([buffer contents]) + offset, &scalar, sizeof(scalar));
}

MTLLanguageVersion latest_stable_msl_version() {
  // Select the newest language revision exposed by the build SDK and host OS.
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

int64_t compile_library(const std::string& source, bool fast_math) {
  @autoreleasepool {
    id<MTLDevice> device = at::mps::getCurrentMPSStream()->device();
    NSString* metal_source = [NSString stringWithUTF8String:source.c_str()];
    NSError* error = nil;
    MTLCompileOptions* compile_options =
        [[[MTLCompileOptions alloc] init] autorelease];
    // Avoid the unreliable implicit default in a Torch JIT extension.  The
    // kernels require at least MSL 3.0 for atomic_float.
    compile_options.languageVersion = latest_stable_msl_version();
    // Metal compiles with fast math unless told otherwise; physics kernels
    // follow HydroForge's math mode instead.
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
      compile_options.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
      compile_options.mathFloatingPointFunctions =
          fast_math ? MTLMathFloatingPointFunctionsFast
                    : MTLMathFloatingPointFunctionsPrecise;
    } else {
#endif
      compile_options.fastMathEnabled = fast_math;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    }
#endif
    // The registry owns the +1 reference for the life of the process.
    id<MTLLibrary> library = [device newLibraryWithSource:metal_source
                                                  options:compile_options
                                                    error:&error];
    TORCH_CHECK(library != nil, "Metal library compilation failed: ",
                error ? error.localizedDescription.UTF8String : "unknown error");
    std::lock_guard<std::mutex> guard(library_mutex);
    libraries.push_back(library);
    return static_cast<int64_t>(libraries.size() - 1);
  }
}

id<MTLLibrary> get_library(int64_t library_id) {
  std::lock_guard<std::mutex> guard(library_mutex);
  TORCH_CHECK(library_id >= 0 &&
                  static_cast<size_t>(library_id) < libraries.size(),
              "Invalid Metal library id: ", library_id);
  return libraries[static_cast<size_t>(library_id)];
}

int64_t create_pipeline(
    int64_t library_id,
    const std::string& kernel_name,
    const std::vector<std::tuple<uint32_t, std::string, double>>& function_constants,
    const std::vector<std::string>& argument_types,
    const std::vector<std::string>& argument_access) {
  @autoreleasepool {
    TORCH_CHECK(argument_types.size() == argument_access.size(),
                "Metal argument access/type count mismatch");
    for (size_t i = 0; i < argument_types.size(); ++i) {
      const auto& kind = argument_types[i];
      const auto& access = argument_access[i];
      TORCH_CHECK(kind == "buffer" || kind == "float32" || kind == "int32" ||
                  kind == "uint32" || kind == "int64" || kind == "bool",
                  "Unsupported Metal argument type: ", kind);
      TORCH_CHECK(kind == "buffer"
                  ? (access == "read" || access == "write" || access == "read_write")
                  : access == "none", "Metal argument kind/access mismatch");
    }
    std::unordered_set<uint32_t> indices;
    for (const auto& [index, type, value] : function_constants) {
      TORCH_CHECK(indices.insert(index).second, "Duplicate Metal function constant index");
      TORCH_CHECK(std::isfinite(value), "Metal function constants must be finite");
      if (type == "bool") {
        TORCH_CHECK(value == 0.0 || value == 1.0, "Metal bool constant must be 0 or 1");
      } else if (type == "int32") {
        TORCH_CHECK(std::trunc(value) == value && value >= INT32_MIN && value <= INT32_MAX,
                    "Metal int32 constant must be an in-range integer");
      } else if (type == "float32") {
        TORCH_CHECK(std::abs(value) <= std::numeric_limits<float>::max() &&
                    (value == 0.0 || static_cast<float>(value) != 0.0f),
                    "Metal float32 constant overflows or underflows");
      } else {
        TORCH_CHECK(false, "Unsupported Metal function constant type: ", type);
      }
    }
    id<MTLDevice> device = at::mps::getCurrentMPSStream()->device();
    id<MTLLibrary> library = get_library(library_id);

    MTLFunctionConstantValues* constants =
        [[[MTLFunctionConstantValues alloc] init] autorelease];
    for (const auto& [index, type, value] : function_constants) {
      if (type == "bool") {
        bool copy = value != 0.0;
        [constants setConstantValue:&copy type:MTLDataTypeBool atIndex:index];
      } else if (type == "int32") {
        int32_t copy = static_cast<int32_t>(value);
        [constants setConstantValue:&copy type:MTLDataTypeInt atIndex:index];
      } else if (type == "float32") {
        float copy = static_cast<float>(value);
        [constants setConstantValue:&copy type:MTLDataTypeFloat atIndex:index];
      } else {
        TORCH_CHECK(false, "Unsupported Metal function constant type: ", type);
      }
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
    [function autorelease];

    MTLComputePipelineDescriptor* descriptor =
        [[[MTLComputePipelineDescriptor alloc] init] autorelease];
    descriptor.computeFunction = function;
    descriptor.supportIndirectCommandBuffers = YES;
    NSError* pipeline_error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithDescriptor:descriptor
                                               options:MTLPipelineOptionNone
                                            reflection:nil
                                                 error:&pipeline_error];
    [pipeline autorelease];
    TORCH_CHECK(pipeline != nil, "Metal pipeline creation failed: ",
                pipeline_error
                    ? pipeline_error.localizedDescription.UTF8String
                    : "unknown error");
    id<MTLArgumentEncoder> argument_encoder =
        [function newArgumentEncoderWithBufferIndex:0];
    [argument_encoder autorelease];
    TORCH_CHECK(argument_encoder != nil,
                "Failed to create Metal argument encoder for ", kernel_name);
    std::lock_guard<std::mutex> guard(pipeline_mutex);
    std::vector<Pipeline::ArgumentType> encoded_types;
    encoded_types.reserve(argument_types.size());
    for (const auto& kind : argument_types) {
      if (kind == "buffer") encoded_types.push_back(Pipeline::ArgumentType::Buffer);
      else if (kind == "float32") encoded_types.push_back(Pipeline::ArgumentType::Float32);
      else if (kind == "int32") encoded_types.push_back(Pipeline::ArgumentType::Int32);
      else if (kind == "uint32") encoded_types.push_back(Pipeline::ArgumentType::UInt32);
      else if (kind == "int64") encoded_types.push_back(Pipeline::ArgumentType::Int64);
      else if (kind == "bool") encoded_types.push_back(Pipeline::ArgumentType::Bool);
      else TORCH_CHECK(false, "Unsupported Metal argument type: ", kind);
    }
    auto item = std::make_shared<Pipeline>();
    item->state = [pipeline retain];
    item->argument_encoder = [argument_encoder retain];
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

std::shared_ptr<ArgumentBinding> get_binding(int64_t binding_id, int64_t pipeline_id) {
  std::lock_guard<std::mutex> guard(binding_mutex);
  auto found = bindings.find(binding_id);
  TORCH_CHECK(found != bindings.end(),
              "Invalid or released Metal argument binding id: ", binding_id);
  auto binding = found->second;
  TORCH_CHECK(binding->pipeline_id == pipeline_id,
              "Metal argument binding belongs to a different pipeline");
  return binding;
}

void release_argument_binding(int64_t binding_id) {
  std::shared_ptr<ArgumentBinding> binding;
  {
    std::lock_guard<std::mutex> guard(binding_mutex);
    auto found = bindings.find(binding_id);
    if (found == bindings.end()) return;
    binding = std::move(found->second);
    bindings.erase(found);
  }
  drain_retired_resources();
  // An ICB keeps its own reference; only direct encodes need a GPU fence.
  const EncodedOn encoded_on = binding->encoded_on;
  release_after_completion(std::move(binding), encoded_on);
}

int64_t create_argument_binding(
    int64_t pipeline_id, const pybind11::list& arguments) {
  drain_retired_resources();
  auto pipeline = get_pipeline(pipeline_id);
  TORCH_CHECK(arguments.size() == pipeline->argument_types.size(),
              "Metal argument/type count mismatch");
  id<MTLDevice> device = at::mps::getCurrentMPSStream()->device();
  auto binding = std::make_shared<ArgumentBinding>();
  binding->pipeline_id = pipeline_id;
  binding->encoded = [device
      newBufferWithLength:pipeline->argument_encoder.encodedLength
                  options:MTLResourceStorageModeShared];
  TORCH_CHECK(binding->encoded != nil,
              "Failed to allocate Metal argument buffer");
  [pipeline->argument_encoder setArgumentBuffer:binding->encoded offset:0];
  binding->resources.push_back({binding->encoded, MTLResourceUsageRead});
  NSUInteger scalar_count = 0;
  for (const auto kind : pipeline->argument_types) {
    if (kind != Pipeline::ArgumentType::Buffer) ++scalar_count;
  }
  id<MTLBuffer> scalars = nil;
  if (scalar_count != 0) {
    scalars = [device newBufferWithLength:scalar_count * kScalarSlot
                                  options:MTLResourceStorageModeShared];
    TORCH_CHECK(scalars != nil, "Failed to allocate Metal scalar argument buffer");
    // The binding owns the +1 reference from here on, including on errors.
    binding->owned_scalar_buffers.push_back(scalars);
    binding->resources.push_back({scalars, MTLResourceUsageRead});
  }
  NSUInteger scalar_offset = 0;
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
    } else {
      buffer = scalars;
      offset = scalar_offset;
      scalar_offset += kScalarSlot;
      if (kind == Pipeline::ArgumentType::Float32) {
        write_scalar<float>(buffer, offset, value);
      } else if (kind == Pipeline::ArgumentType::Int32) {
        write_scalar<int32_t>(buffer, offset, value);
      } else if (kind == Pipeline::ArgumentType::UInt32) {
        write_scalar<uint32_t>(buffer, offset, value);
      } else if (kind == Pipeline::ArgumentType::Int64) {
        write_scalar<int64_t>(buffer, offset, value);
      } else {
        write_scalar<bool>(buffer, offset, value);
      }
    }
    [pipeline->argument_encoder setBuffer:buffer offset:offset atIndex:i];
    if (buffer != nil && kind == Pipeline::ArgumentType::Buffer) {
      binding->resources.push_back({
          buffer, pipeline->argument_usage[static_cast<size_t>(i)]});
    }
  }
  std::lock_guard<std::mutex> guard(binding_mutex);
  TORCH_CHECK(next_binding_id < std::numeric_limits<int64_t>::max(),
              "Metal argument binding IDs exhausted");
  const int64_t id = next_binding_id++;
  bindings.emplace(id, std::move(binding));
  return id;
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
  drain_retired_resources();
  validate_grid_extent(threads);
  auto pipeline = get_pipeline(pipeline_id);
  auto binding = get_binding(binding_id, pipeline_id);
  if (threads == 0) return;

  auto* stream = at::mps::getCurrentMPSStream();
  binding->encoded_on.note(stream);
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline->state];
    [encoder setBuffer:binding->encoded offset:0 atIndex:0];
    for (const auto& [resource, usage] : binding->resources) {
      [encoder useResource:resource usage:usage];
    }
    NSUInteger width = validate_group_size(*pipeline, requested_group_size);
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
  drain_retired_resources();
  const size_t count = pipeline_ids.size();
  TORCH_CHECK(binding_ids.size() == count && threads.size() == count &&
                  group_sizes.size() == count && barriers.size() == count,
              "Metal sequence arrays must have equal length");
  std::vector<std::shared_ptr<Pipeline>> command_pipelines;
  std::vector<std::shared_ptr<ArgumentBinding>> command_bindings;
  for (size_t i = 0; i < count; ++i) {
    validate_grid_extent(threads[i]);
    auto pipeline = get_pipeline(pipeline_ids[i]);
    validate_group_size(*pipeline, group_sizes[i]);
    command_pipelines.push_back(pipeline);
    command_bindings.push_back(get_binding(binding_ids[i], pipeline_ids[i]));
  }
  auto* stream = at::mps::getCurrentMPSStream();
  for (size_t i = 0; i < count; ++i) {
    if (threads[i] != 0) command_bindings[i]->encoded_on.note(stream);
  }
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    for (size_t i = 0; i < count; ++i) {
      if (threads[i] == 0) continue;
      auto pipeline = command_pipelines[i];
      auto binding = command_bindings[i];
      [encoder setComputePipelineState:pipeline->state];
      [encoder setBuffer:binding->encoded offset:0 atIndex:0];
      for (const auto& [resource, usage] : binding->resources) {
        [encoder useResource:resource usage:usage];
      }
      NSUInteger width = validate_group_size(*pipeline, group_sizes[i]);
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
  std::vector<std::shared_ptr<Pipeline>> command_pipelines;
  std::vector<std::shared_ptr<ArgumentBinding>> command_bindings;
  for (size_t i = 0; i < count; ++i) {
    validate_grid_extent(threads[i]);
    auto pipeline = get_pipeline(pipeline_ids[i]);
    validate_group_size(*pipeline, group_sizes[i]);
    command_pipelines.push_back(pipeline);
    command_bindings.push_back(get_binding(binding_ids[i], pipeline_ids[i]));
  }
  for (uint64_t extent : threads) {
    TORCH_CHECK(extent > 0, "ICB dispatch dimensions must be positive");
  }
  auto* stream = at::mps::getCurrentMPSStream();
  id<MTLDevice> device = stream->device();
  MTLIndirectCommandBufferDescriptor* descriptor =
      [[MTLIndirectCommandBufferDescriptor alloc] init];
  descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatchThreads;
  descriptor.inheritPipelineState = NO;
  descriptor.inheritBuffers = NO;
  descriptor.maxKernelBufferBindCount = 1;
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
    auto pipeline = command_pipelines[command_index];
    auto binding = command_bindings[command_index];
    id<MTLIndirectComputeCommand> command =
        [commands indirectComputeCommandAtIndex:command_index];
    [command setComputePipelineState:pipeline->state];
    [command setKernelBuffer:binding->encoded offset:0 atIndex:0];
    graph->retained_bindings.push_back(binding);
    for (const auto& [resource, usage] : binding->resources) {
      add_resource(resource, usage);
    }
    NSUInteger width = validate_group_size(*pipeline, group_sizes[command_index]);
    [command concurrentDispatchThreads:MTLSizeMake(threads[command_index], 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    // MTLIndirectComputeCommand::setBarrier waits for commands *before* the
    // command it is attached to.  The Python/native sequence ABI records a
    // barrier *after* command i (matching dispatch_sequence above), so shift
    // the marker forward by one command.  Wrapping the final marker onto the
    // first command also preserves the dependency between repeated ICB
    // executions used by fixed/adaptive substep loops.
    const size_t preceding_command_index =
        (command_index + count - 1) % count;
    if (barriers[preceding_command_index]) [command setBarrier];
  }
  std::lock_guard<std::mutex> guard(graph_mutex);
  TORCH_CHECK(next_graph_id < std::numeric_limits<int64_t>::max(),
              "Metal ICB graph IDs exhausted");
  const int64_t id = next_graph_id++;
  graphs.emplace(id, std::move(graph));
  return id;
}

void replay_icb(int64_t graph_id, uint64_t replays) {
  drain_retired_resources();
  TORCH_CHECK(replays > 0, "ICB replay count must be positive");
  std::shared_ptr<ICBGraph> graph;
  {
    std::lock_guard<std::mutex> guard(graph_mutex);
    auto found = graphs.find(graph_id);
    TORCH_CHECK(found != graphs.end(), "Invalid or released Metal ICB graph id: ", graph_id);
    graph = found->second;
  }
  auto* stream = at::mps::getCurrentMPSStream();
  graph->encoded_on.note(stream);
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
    auto found = graphs.find(graph_id);
    if (found == graphs.end()) return;
    graph = std::move(found->second);
    graphs.erase(found);
  }
  // Destroy buffers outside the registry lock. Releasing the same id is safe.
  drain_retired_resources();
  const EncodedOn encoded_on = graph->encoded_on;
  release_after_completion(std::move(graph), encoded_on);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("compile_library", &compile_library);
  module.def("create_pipeline", &create_pipeline);
  module.def("create_argument_binding", &create_argument_binding);
  module.def("release_argument_binding", &release_argument_binding);
  module.def("dispatch", &dispatch);
  module.def("dispatch_sequence", &dispatch_sequence);
  module.def("create_icb", &create_icb);
  module.def("replay_icb", &replay_icb, pybind11::arg("graph_id"),
             pybind11::arg("replays") = 1);
  module.def("release_icb", &release_icb);
}
