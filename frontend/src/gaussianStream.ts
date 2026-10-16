// A cancellable, memory-bounded ReadableStream over a local Blob.
//
// Spark's SplatLoader pulls one chunk at a time from the stream it is given and
// hands each chunk straight to its WebAssembly ChunkDecoder in a worker, so the
// only bytes on the main thread at any moment are the current chunk. Wrapping
// Blob.stream() rather than passing it directly buys two things Spark does not
// offer: an exact count of the bytes that have actually left the file, and a
// cancel() that stops feeding Spark without rejecting a read Spark is awaiting.
//
// Cancellation deliberately closes the stream instead of erroring it. Spark
// treats a zero-length chunk as end-of-input, so a cancelled import unwinds
// through Spark's normal completion path; the caller then discards whatever was
// produced. Erroring the stream would reject inside Spark's own status callback,
// which is not awaited, and would surface as an unhandled rejection.

export interface BoundedStream {
  stream: ReadableStream<Uint8Array>;
  /** Bytes pulled out of the file so far. */
  readBytes: () => number;
  /** Stop feeding Spark and release the underlying file reader. */
  cancel: () => void;
}

export function boundedStream(blob: Blob): BoundedStream {
  const reader = blob.stream().getReader();
  let bytes = 0;
  let cancelled = false;
  let released = false;

  const release = () => {
    if (released) return;
    released = true;
    void reader.cancel().catch(() => {});
  };

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (cancelled) {
        release();
        controller.close();
        return;
      }
      const { done, value } = await reader.read();
      if (done || cancelled || !value) {
        release();
        controller.close();
        return;
      }
      bytes += value.byteLength;
      controller.enqueue(value);
    },
    cancel() {
      cancelled = true;
      release();
    },
  });

  return {
    stream,
    readBytes: () => bytes,
    cancel: () => {
      cancelled = true;
      release();
    },
  };
}
