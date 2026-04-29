// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "DiffusionMetal",
    platforms: [.macOS(.v11)],
    products: [
        .executable(name: "DiffusionMetal", targets: ["DiffusionMetal"])
    ],
    targets: [
        .executableTarget(
            name: "DiffusionMetal",
            resources: [
                .copy("Shaders.metal")
            ]
        )
    ]
)
