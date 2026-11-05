/** Upload a host view in bounded pieces. Dawn/Metal (webgpu 0.6.0) aborts in
 * BufferBase::UploadData when one writeBuffer above its 4 MiB staging ring
 * has an odd 4-byte word count: the dedicated staging reservation is rounded
 * and its size check fails. 1 MiB pieces stay inside the ring and bound
 * staging memory; the bytes written are identical. */
export const UPLOAD_CHUNK_BYTES=1<<20;
export function writeBufferChunked(queue:GPUQueue,buffer:GPUBuffer,offset:number,data:ArrayBufferView):void{
 if(data.byteLength%4)throw new Error('GPU uploads must be whole 4-byte words');
 for(let from=0;from<data.byteLength;from+=UPLOAD_CHUNK_BYTES)
  queue.writeBuffer(buffer,offset+from,data.buffer as ArrayBuffer,data.byteOffset+from,Math.min(UPLOAD_CHUNK_BYTES,data.byteLength-from));
}
