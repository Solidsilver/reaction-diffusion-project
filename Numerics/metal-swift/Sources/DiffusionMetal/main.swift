import AppKit
import Metal
import MetalKit

let WIDTH = 1920
let HEIGHT = 1080
let SUBSTEPS = 100
let DEPTH = 2

enum Seed {
    case rect(x: Int, y: Int, width: Int, height: Int, b: Float)
    case circle(cx: Int, cy: Int, radius: Int, b: Float)
}

func generateInitialState(seeds: [Seed]) -> [Float] {
    var data = [Float](repeating: 0.0, count: HEIGHT * WIDTH * DEPTH)

    // Fill A channel with 1.0 everywhere
    for y in 0..<HEIGHT {
        for x in 0..<WIDTH {
            let aIdx = (y * WIDTH + x) * DEPTH + 0
            data[aIdx] = 1.0
        }
    }

    // Apply seeds to B channel
    for seed in seeds {
        switch seed {
        case .rect(let rx, let ry, let rw, let rh, let b):
            for y in max(0, ry)..<min(HEIGHT, ry + rh) {
                for x in max(0, rx)..<min(WIDTH, rx + rw) {
                    let bIdx = (y * WIDTH + x) * DEPTH + 1
                    data[bIdx] = b
                }
            }
        case .circle(let cx, let cy, let r, let b):
            let rSq = r * r
            for y in max(0, cy - r)..<min(HEIGHT, cy + r) {
                for x in max(0, cx - r)..<min(WIDTH, cx + r) {
                    let dx = x - cx
                    let dy = y - cy
                    if dx * dx + dy * dy <= rSq {
                        let bIdx = (y * WIDTH + x) * DEPTH + 1
                        data[bIdx] = b
                    }
                }
            }
        }
    }

    return data
}

func writeInitialState(data: [Float], to buffer: MTLBuffer) {
    data.withUnsafeBytes { bytes in
        buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var mtkView: MTKView!
    var renderer: Renderer!

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        let frame = NSRect(x: 0, y: 0, width: WIDTH, height: HEIGHT)
        mtkView = MTKView(frame: frame, device: device)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.framebufferOnly = false
        mtkView.autoResizeDrawable = false
        mtkView.drawableSize = CGSize(width: WIDTH, height: HEIGHT)

        renderer = Renderer(view: mtkView, device: device)
        mtkView.delegate = renderer

        window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Reaction-Diffusion"
        window.contentView = mtkView
        window.makeKeyAndOrderFront(nil)
        window.center()

        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            if event.keyCode == 49 {  // Space
                self.renderer.paused.toggle()
            }
            return event
        }

        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    let updatePipeline: MTLComputePipelineState
    let fillPixelsPipeline: MTLComputePipelineState

    let curBuffer: MTLBuffer
    let prevBuffer: MTLBuffer
    let pixelTexture: MTLTexture

    var currentSimBuffer: MTLBuffer
    var previousSimBuffer: MTLBuffer

    var paused = true

    init(view: MTKView, device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        let bufferSize = HEIGHT * WIDTH * 2 * MemoryLayout<Float>.size
        self.curBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        self.prevBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        self.currentSimBuffer = curBuffer
        self.previousSimBuffer = prevBuffer

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: WIDTH,
            height: HEIGHT,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        self.pixelTexture = device.makeTexture(descriptor: textureDescriptor)!

        guard let shaderURL = Bundle.module.url(forResource: "Shaders", withExtension: "metal")
        else {
            fatalError("Failed to find Shaders.metal in bundle")
        }
        let shaderSource = try! String(contentsOf: shaderURL)
        let library = try! device.makeLibrary(source: shaderSource, options: nil)

        let updateFunction = library.makeFunction(name: "update")!
        self.updatePipeline = try! device.makeComputePipelineState(function: updateFunction)

        let fillPixelsFunction = library.makeFunction(name: "fill_pixels")!
        self.fillPixelsPipeline = try! device.makeComputePipelineState(function: fillPixelsFunction)

        super.init()

        let seeds: [Seed] = [
            // Center rectangle (original default)
            .rect(x: WIDTH / 2 - 100, y: HEIGHT / 2 - 200, width: 200, height: 400, b: 1.0),
            // Extra small circles for fun
            .circle(cx: 400, cy: 400, radius: 30, b: 1.0),
            .circle(cx: WIDTH - 400, cy: HEIGHT - 400, radius: 40, b: 1.0),
        ]

        let initialData = generateInitialState(seeds: seeds)
        writeInitialState(data: initialData, to: curBuffer)
        writeInitialState(data: initialData, to: prevBuffer)
        runFillPixels(buffer: prevBuffer)
    }

    private func dispatchSize() -> (threadsPerThreadgroup: MTLSize, threadgroups: MTLSize) {
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (WIDTH + 15) / 16,
            height: (HEIGHT + 15) / 16,
            depth: 1
        )
        return (threadsPerThreadgroup, threadgroups)
    }

    func runFillPixels(buffer: MTLBuffer) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return }

        encoder.setComputePipelineState(fillPixelsPipeline)
        encoder.setTexture(pixelTexture, index: 0)
        encoder.setBuffer(buffer, offset: 0, index: 0)

        let (threads, groups) = dispatchSize()
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
            let commandBuffer = commandQueue.makeCommandBuffer()
        else { return }

        if !paused {
            var writeBuffer = currentSimBuffer
            var readBuffer = previousSimBuffer
            for _ in 0..<SUBSTEPS {
                if let encoder = commandBuffer.makeComputeCommandEncoder() {
                    encoder.setComputePipelineState(updatePipeline)
                    encoder.setBuffer(writeBuffer, offset: 0, index: 0)
                    encoder.setBuffer(readBuffer, offset: 0, index: 1)

                    let (threads, groups) = dispatchSize()
                    encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
                    encoder.endEncoding()
                }
                swap(&writeBuffer, &readBuffer)
            }
            currentSimBuffer = readBuffer
            previousSimBuffer = writeBuffer

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(fillPixelsPipeline)
                encoder.setTexture(pixelTexture, index: 0)
                encoder.setBuffer(currentSimBuffer, offset: 0, index: 0)

                let (threads, groups) = dispatchSize()
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
                encoder.endEncoding()
            }
        }

        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.copy(from: pixelTexture, to: drawable.texture)
            blitEncoder.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
