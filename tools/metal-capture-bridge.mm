// Optional macOS diagnostic addon, never loaded by the application.
// Build with the headers for the Node executable running the worker:
// xcrun clang++ -std=c++17 -fobjc-arc -shared -fPIC -undefined dynamic_lookup \
//   -I /path/to/node/include/node -framework Metal -framework Foundation \
//   tools/metal-capture-bridge.mm -o /tmp/fluid-metal-capture.node
// Start Node with MTL_CAPTURE_ENABLED=1 and FLUID_METAL_CAPTURE_BRIDGE set to
// that addon. FLUID_METAL_CAPTURE_FRAME selects one physical simulation frame.
#include <node_api.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string>

static napi_value Start(napi_env env, napi_callback_info info) {
  size_t count = 1;
  napi_value argument;
  size_t length = 0;
  if (napi_get_cb_info(env, info, &count, &argument, nullptr, nullptr) != napi_ok ||
      count != 1 || napi_get_value_string_utf8(env, argument, nullptr, 0, &length) != napi_ok) {
    napi_throw_type_error(env, nullptr, "start requires a capture output path");
    return nullptr;
  }
  std::string path(length + 1, '\0');
  if (napi_get_value_string_utf8(env, argument, path.data(), path.size(), &length) != napi_ok) {
    napi_throw_type_error(env, nullptr, "invalid capture output path");
    return nullptr;
  }
  @autoreleasepool {
    MTLCaptureManager *manager = [MTLCaptureManager sharedCaptureManager];
    if (manager.isCapturing) {
      napi_throw_error(env, nullptr, "a Metal capture is already active");
      return nullptr;
    }
    MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.captureObject = MTLCreateSystemDefaultDevice();
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    NSError *error = nil;
    if (![manager startCaptureWithDescriptor:descriptor error:&error]) {
      napi_throw_error(env, nullptr, error.localizedDescription.UTF8String);
      return nullptr;
    }
  }
  napi_value result;
  napi_get_boolean(env, true, &result);
  return result;
}

static napi_value Stop(napi_env env, napi_callback_info info) {
  @autoreleasepool { [[MTLCaptureManager sharedCaptureManager] stopCapture]; }
  napi_value result;
  napi_get_undefined(env, &result);
  return result;
}

static napi_value Init(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    {"start", nullptr, Start, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"stop", nullptr, Stop, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_define_properties(env, exports, 2, methods);
  return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
