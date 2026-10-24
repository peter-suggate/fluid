/**
 * A Metal shader cache of the headless harness's own (macOS only).
 *
 * Metal persists compiled libraries and pipeline binaries in the user cache
 * directory (com.apple.metal/), but a process reads and writes only one shard
 * of it: the lowest-numbered shard no other process holds. A process holds its
 * shard while it compiles and for some seconds after. Every unbundled Metal
 * process of the user (all of node) shares that directory, so a solver built
 * while any other node process compiles lands on another shard, finds none of
 * the pipelines a previous run stored, and stores its own where the next run
 * is unlikely to look: a repeat construction is as cold as the first. Measured
 * here with two processes: a kernel compiled by one while another compiled
 * was a miss again on the next run, alone; compiled alone, it was a hit.
 *
 * MTLSetShaderCachePath (Metal.framework, exported) moves the process's cache
 * to a directory of its own, before Metal opens one. The harness's directory
 * is shared only by harness processes, so a construction finds the shard its
 * predecessors filled unless another harness process is compiling right then.
 *
 * The call needs native code: a one-function N-API module, compiled on first use with
 * the Command Line Tools' clang and kept beside the cache. Without clang, or
 * off macOS, nothing changes: Metal keeps the system cache.
 *
 * FLUID_METAL_SHADER_CACHE=<directory> picks the directory; =system keeps the
 * system cache. The default is <user cache dir>/fluid-headless/metal-shader-cache,
 * the user cache directory being the one Metal itself uses
 * (getconf DARWIN_USER_CACHE_DIR, /var/folders/../C).
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

/** Plain C plus one Objective-C call: N-API is declared here rather than
 * taken from node's headers, which an nvm install need not carry. */
const ADDON_SOURCE = String.raw`#import <Foundation/Foundation.h>
#include <stddef.h>
typedef struct napi_env__* napi_env;
typedef struct napi_value__* napi_value;
typedef struct napi_callback_info__* napi_callback_info;
typedef int napi_status;
typedef napi_value (*napi_callback)(napi_env, napi_callback_info);
extern napi_status napi_get_cb_info(napi_env, napi_callback_info, size_t*, napi_value*, napi_value*, void**);
extern napi_status napi_get_value_string_utf8(napi_env, napi_value, char*, size_t, size_t*);
extern napi_status napi_create_function(napi_env, const char*, size_t, napi_callback, void*, napi_value*);
extern napi_status napi_set_named_property(napi_env, napi_value, const char*, napi_value);
extern napi_status napi_get_undefined(napi_env, napi_value*);
extern void MTLSetShaderCachePath(NSString*);
static napi_value setShaderCachePath(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value argv[1]; char path[4096]; size_t length = 0; napi_value result;
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  napi_get_undefined(env, &result);
  if (argc < 1 || napi_get_value_string_utf8(env, argv[0], path, sizeof path, &length) != 0) return result;
  @autoreleasepool { MTLSetShaderCachePath([NSString stringWithUTF8String:path]); }
  return result;
}
napi_value napi_register_module_v1(napi_env env, napi_value exports) {
  napi_value f;
  napi_create_function(env, "setShaderCachePath", (size_t)-1, setShaderCachePath, NULL, &f);
  napi_set_named_property(env, exports, "setShaderCachePath", f);
  return exports;
}
`;

interface MetalShaderCacheAddon { setShaderCachePath(path: string): void }

export interface MetalShaderCacheState {
  /** The directory Metal was pointed at, or undefined for the system cache. */
  readonly directory?: string;
  /** Why the system cache was kept, when it was. */
  readonly reason?: string;
}

let state: MetalShaderCacheState | undefined;

/** Metal's own root: getconf DARWIN_USER_CACHE_DIR, else ~/Library/Caches. */
function userCacheDirectory(): string {
  try { const d = execFileSync("getconf", ["DARWIN_USER_CACHE_DIR"], { encoding: "utf8" }).trim(); if (d) return d; } catch { /* below */ }
  return join(process.env.HOME ?? "/tmp", "Library", "Caches");
}

/** Build (once per source revision) and load the addon from `root`. */
function loadAddon(root: string): MetalShaderCacheAddon {
  const hash = createHash("sha256").update(ADDON_SOURCE).update(process.arch).digest("hex").slice(0, 12);
  const binary = join(root, `metal-shader-cache-${hash}.node`);
  if (!existsSync(binary)) {
    mkdirSync(root, { recursive: true });
    const unique = `${process.pid}-${Date.now().toString(36)}`, source = join(root, `addon-${unique}.m`), built = join(root, `addon-${unique}.node`);
    try {
      writeFileSync(source, ADDON_SOURCE);
      execFileSync("xcrun", ["clang", "-bundle", "-undefined", "dynamic_lookup", "-fobjc-arc", "-O2",
        "-framework", "Foundation", "-framework", "Metal", "-o", built, source], { stdio: ["ignore", "ignore", "pipe"] });
      renameSync(built, binary); // Atomic: a concurrent builder's rename lands the same bytes.
    } finally { rmSync(source, { force: true }); rmSync(built, { force: true }); }
  }
  return createRequire(import.meta.url)(binary) as MetalShaderCacheAddon;
}

/**
 * Point this process's Metal shader cache at the harness directory. Call
 * before the first Metal device exists (before dawn.create): Metal ignores
 * the path once its cache is open. Idempotent; never throws.
 */
export function configureMetalShaderCache(): MetalShaderCacheState {
  if (state) return state;
  const requested = process.env.FLUID_METAL_SHADER_CACHE?.trim();
  if (process.platform !== "darwin") return state = { reason: "not macOS" };
  if (requested === "system" || requested === "off" || requested === "0") return state = { reason: `FLUID_METAL_SHADER_CACHE=${requested}` };
  try {
    const root = join(userCacheDirectory(), "fluid-headless");
    const addon = loadAddon(root);
    const directory = requested || join(root, "metal-shader-cache");
    mkdirSync(directory, { recursive: true });
    addon.setShaderCachePath(directory);
    return state = { directory };
  } catch (error) {
    const reason = error instanceof Error ? error.message.split("\n")[0]! : String(error);
    console.warn(`[metal-shader-cache] keeping the system Metal shader cache: ${reason}`);
    return state = { reason };
  }
}
