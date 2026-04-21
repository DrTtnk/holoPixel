import { pathtracerVS, pathtracerFS, fringeFS, fftFS, accumulatorFS, displayFS } from './shaders.ts';

export class SimulationEngine {
    private gl: WebGL2RenderingContext;
    private width: number;
    private height: number;

    private programs: { [key: string]: WebGLProgram } = {};
    private quadVao: WebGLVertexArrayObject | null = null;
    private fbos: { [key: string]: { fbo: WebGLFramebuffer, tex: WebGLTexture } } = {};

    private N = 512; // Frequency / Subpixel array width/height (power of 2)
    private lightFieldSize = 64; // Hemisphere render target size

    // Simulation State
    public running = false;
    private targetHogelsX = 800; // Physical Screen configuration
    private targetHogelsY = 600;
    
    // We will use an array of coordinates to process in a spiral
    private processingOrder: {x: number, y: number}[] = [];
    public currentHogelIndex = 0;
    public totalHogelsComputed = 0;

    // Diagnostics State
    private lastUpdateTime = 0;
    private lastComputedCount = 0;
    private hogelsPerSecond = 0;
    private previewFboWidth = 128;
    
    public onProgressUpdate?: (stats: any) => void;

    constructor(canvas: HTMLCanvasElement) {
        // Need EXT_color_buffer_float for HDR accumulation
        const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
        if (!gl) throw new Error("WebGL2 not supported.");
        const ext = gl.getExtension('EXT_color_buffer_float');
        if(!ext) console.warn("EXT_color_buffer_float not supported! Float accumulation may fail.");
        
        this.gl = gl;
        this.width = canvas.width;
        this.height = canvas.height;

        this.initPrograms();
        this.initQuad();
        this.initFBOs();
        this.generateSpiralOrder();
        
        this.lastUpdateTime = performance.now();
    }

    private generateSpiralOrder() {
        const cx = Math.floor(this.targetHogelsX / 2);
        const cy = Math.floor(this.targetHogelsY / 2);
        
        let x = cx;
        let y = cy;
        let dx = 1;
        let dy = 0;
        let segmentLength = 1;
        let segmentPassed = 0;
        
        const total = this.targetHogelsX * this.targetHogelsY;
        
        // Generate enough points to cover the screen (the spiral goes out of bounds, so we loop until we collect 'total' valid points)
        while (this.processingOrder.length < total) {
            if (x >= 0 && x < this.targetHogelsX && y >= 0 && y < this.targetHogelsY) {
                this.processingOrder.push({ x, y });
            }
            
            x += dx;
            y += dy;
            segmentPassed++;
            
            if (segmentPassed === segmentLength) {
                segmentPassed = 0;
                // Turn 90 degrees right
                const temp = dx;
                dx = -dy;
                dy = temp;
                
                // Increase segment length every time we complete a horizontal movement
                if (dy === 0) {
                    segmentLength++;
                }
            }
        }
    }

    private compileShader(src: string, type: number): WebGLShader {
        const shader = this.gl.createShader(type)!;
        this.gl.shaderSource(shader, src);
        this.gl.compileShader(shader);
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(shader));
            // Log source code up to error
            console.log(src.split('\\n').map((l, i) => `${i+1}: ${l}`).join('\\n'));
        }
        return shader;
    }

    private createProgram(vsSrc: string, fsSrc: string): WebGLProgram {
        const vs = this.compileShader(vsSrc, this.gl.VERTEX_SHADER);
        const fs = this.compileShader(fsSrc, this.gl.FRAGMENT_SHADER);
        const prog = this.gl.createProgram()!;
        this.gl.attachShader(prog, vs);
        this.gl.attachShader(prog, fs);
        this.gl.linkProgram(prog);
        if (!this.gl.getProgramParameter(prog, this.gl.LINK_STATUS)) {
            console.error(this.gl.getProgramInfoLog(prog));
        }
        return prog;
    }

    private initPrograms() {
        this.programs.pathtracer = this.createProgram(pathtracerVS, pathtracerFS);
        this.programs.fringe = this.createProgram(pathtracerVS, fringeFS);
        this.programs.fft = this.createProgram(pathtracerVS, fftFS);
        this.programs.accumulator = this.createProgram(pathtracerVS, accumulatorFS);
        this.programs.display = this.createProgram(pathtracerVS, displayFS);
    }

    private initQuad() {
        this.quadVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.quadVao);

        const vbo = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vbo);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1
        ]), this.gl.STATIC_DRAW);

        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 2, this.gl.FLOAT, false, 0, 0);
        
        this.gl.bindVertexArray(null);
    }

    private createFBO(w: number, h: number, format: number, type: number) {
        const tex = this.gl.createTexture()!;
        this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, format, w, h, 0, this.gl.RGBA, type, null);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);

        const fbo = this.gl.createFramebuffer()!;
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fbo);
        this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, tex, 0);

        return { fbo, tex };
    }

    private initFBOs() {
        const gl = this.gl;
        this.fbos.lightfield = this.createFBO(this.lightFieldSize, this.lightFieldSize, gl.RGBA16F, gl.HALF_FLOAT);
        this.fbos.ping = this.createFBO(this.N, this.N, gl.RGBA32F, gl.FLOAT);
        this.fbos.pong = this.createFBO(this.N, this.N, gl.RGBA32F, gl.FLOAT);
        this.fbos.sensor = this.createFBO(this.width, this.height, gl.RGBA32F, gl.FLOAT);
        // Standard 8-bit FBO for readPixels diagnostic previews
        this.fbos.preview = this.createFBO(this.previewFboWidth, this.previewFboWidth, gl.RGBA8, gl.UNSIGNED_BYTE);
        
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos.sensor.fbo);
        gl.clearColor(0,0,0,1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    private drawQuad() {
        this.gl.bindVertexArray(this.quadVao);
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        this.gl.bindVertexArray(null);
    }

    public computeNextPatch(patchesPerFrame: number = 2) {
        const gl = this.gl;

        for (let step = 0; step < patchesPerFrame; step++) {
            if (this.currentHogelIndex >= this.processingOrder.length) {
                this.running = false;
                break;
            }

            const currentHogel = this.processingOrder[this.currentHogelIndex];
            let nx = (currentHogel.x / this.targetHogelsX) * 2.0 - 1.0;
            let ny = (currentHogel.y / this.targetHogelsY) * 2.0 - 1.0;
            let hogelPosX = nx * 5.0; 
            let hogelPosY = ny * 3.75 + 5.0; // Shifted +5.0 to be at height center of the 10-unit tall room

            // 1. Raytrace Lightfield for this Hogel
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos.lightfield.fbo);
            gl.viewport(0, 0, this.lightFieldSize, this.lightFieldSize);
            gl.useProgram(this.programs.pathtracer);
            gl.uniform3f(gl.getUniformLocation(this.programs.pathtracer, "u_hogelPos"), hogelPosX, hogelPosY, 0);
            this.drawQuad();

            // 2. Compute Fringes
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos.ping.fbo);
            gl.viewport(0, 0, this.N, this.N);
            gl.useProgram(this.programs.fringe);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.fbos.lightfield.tex);
            gl.uniform1i(gl.getUniformLocation(this.programs.fringe, "u_lightfield"), 0);
            this.drawQuad();

            // 3. Execute 2D FFT
            let currentFBO = this.executeFFT();

            // 4. Optical Reconstruction Accumulation
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos.sensor.fbo);
            gl.viewport(0, 0, this.width, this.height);
            gl.useProgram(this.programs.accumulator);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE);
            
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, currentFBO.tex);
            gl.uniform1i(gl.getUniformLocation(this.programs.accumulator, "u_tex"), 0);
            gl.uniform2f(gl.getUniformLocation(this.programs.accumulator, "u_hogelCenter"), nx, ny);
            gl.uniform1f(gl.getUniformLocation(this.programs.accumulator, "u_intensity"), 1.0);
            
            this.drawQuad();
            gl.disable(gl.BLEND);

            this.currentHogelIndex++;
            this.totalHogelsComputed++;
        }
        
        const now = performance.now();
        if (now - this.lastUpdateTime >= 1000) {
            this.hogelsPerSecond = this.totalHogelsComputed - this.lastComputedCount;
            this.lastComputedCount = this.totalHogelsComputed;
            this.lastUpdateTime = now;

            // 5. Draw Sensor to Screen with Tonemap
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, this.width, this.height);
            gl.useProgram(this.programs.display);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.fbos.sensor.tex);
            gl.uniform1i(gl.getUniformLocation(this.programs.display, "u_tex"), 0);
            this.drawQuad();

            // Extract Previews
            const lightfieldData = this.readPreview(this.fbos.lightfield.tex);
            const fringeData = this.readPreview(this.fbos.ping.tex);

            // Broadcast stats
            // Make sure we securely access array to prevent OOB crash at the very end
            const broadcastHogel = this.processingOrder[Math.min(this.currentHogelIndex, this.processingOrder.length - 1)];
            
            if (this.onProgressUpdate) {
                this.onProgressUpdate({
                    computed: this.totalHogelsComputed,
                    total: this.targetHogelsX * this.targetHogelsY,
                    subpixels: this.totalHogelsComputed * this.N * this.N,
                    currentX: broadcastHogel.x,
                    currentY: broadcastHogel.y,
                    hogelsPerSecond: this.hogelsPerSecond,
                    previews: {
                        width: this.previewFboWidth,
                        height: this.previewFboWidth,
                        lightfield: lightfieldData,
                        fringe: fringeData
                    }
                });
            }
        }
    }

    private readPreview(sourceTex: WebGLTexture): Uint8ClampedArray {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos.preview.fbo);
        gl.viewport(0, 0, this.previewFboWidth, this.previewFboWidth);
        gl.useProgram(this.programs.display); // Use display to tonemap it to 0-255
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, sourceTex);
        gl.uniform1i(gl.getUniformLocation(this.programs.display, "u_tex"), 0);
        this.drawQuad();

        const data = new Uint8Array(this.previewFboWidth * this.previewFboWidth * 4);
        gl.readPixels(0, 0, this.previewFboWidth, this.previewFboWidth, gl.RGBA, gl.UNSIGNED_BYTE, data);
        return new Uint8ClampedArray(data.buffer);
    }

    private executeFFT() {
        const gl = this.gl;
        const stages = Math.log2(this.N);
        let ping = true;

        gl.useProgram(this.programs.fft);
        gl.uniform1i(gl.getUniformLocation(this.programs.fft, "u_N"), this.N);

        const runPass = (horizontal: number, stage: number) => {
            const src = ping ? this.fbos.ping : this.fbos.pong;
            const dst = ping ? this.fbos.pong : this.fbos.ping;
            
            gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, src.tex);
            gl.uniform1i(gl.getUniformLocation(this.programs.fft, "u_tex"), 0);
            gl.uniform1i(gl.getUniformLocation(this.programs.fft, "u_stage"), stage);
            gl.uniform1i(gl.getUniformLocation(this.programs.fft, "u_horizontal"), horizontal);
            
            this.drawQuad();
            ping = !ping;
        };

        gl.viewport(0, 0, this.N, this.N);
        // Horizontal
        runPass(1, 0);
        for (let i = 1; i <= stages; i++) runPass(1, i);
        // Vertical
        runPass(0, 0);
        for (let i = 1; i <= stages; i++) runPass(0, i);

        return ping ? this.fbos.ping : this.fbos.pong;
    }
}
