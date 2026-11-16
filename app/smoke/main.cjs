// Five-minute check that the one load-bearing assumption holds: real WebGPU,
// on the real GPU, inside the shell we intend to ship.
const { app, BrowserWindow } = require('electron');

app.whenReady().then(async () => {
  const win = new BrowserWindow({ show: false, width: 800, height: 600 });
  await win.loadFile(require('path').join(__dirname, 'probe.html'));
  const result = await win.webContents.executeJavaScript(`(async () => {
    if (!navigator.gpu) return { ok: false, reason: 'navigator.gpu is undefined',
                                  secure: window.isSecureContext, origin: location.origin };
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { ok: false, reason: 'requestAdapter returned null' };
    const info = adapter.info || {};
    const dev = await adapter.requestDevice();
    // prove compute actually runs, not just that an adapter exists
    const N = 256;
    const out = dev.createBuffer({ size: N*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = dev.createBuffer({ size: N*4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const mod = dev.createShaderModule({ code: \`
      @group(0) @binding(0) var<storage, read_write> o: array<u32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
        o[g.x] = g.x * 3u;
      }\` });
    const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(N/64); pass.end();
    enc.copyBufferToBuffer(out, 0, read, 0, N*4);
    dev.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const got = new Uint32Array(read.getMappedRange().slice(0));
    read.unmap();
    const computeOk = got[7] === 21 && got[255] === 765;
    return {
      ok: true, computeOk,
      vendor: info.vendor, architecture: info.architecture,
      device: info.device, description: info.description,
      isFallback: adapter.isFallbackAdapter === true,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      features: [...adapter.features].slice(0, 12),
    };
  })()`);
  console.log('ELECTRON ' + process.versions.electron + '  CHROMIUM ' + process.versions.chrome);
  console.log(JSON.stringify(result, null, 2));
  app.quit();
});
