import CoreML
import Foundation
import OSLog

enum CoreMLModelLoader {
  private static let logger = Logger(subsystem: "com.pg123521.IsaacItemLens", category: "model")

  static func load(
    _ name: String,
    computeUnits: MLComputeUnits = .all,
    bundle: Bundle = .main
  ) throws -> MLModel {
    guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
      logger.error("missing model name=\(name, privacy: .public) bundle=\(bundle.bundlePath, privacy: .public)")
      throw ModelError.missingModel(name)
    }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    logger.info("loading model name=\(name, privacy: .public) url=\(url.lastPathComponent, privacy: .public) computeUnits=\(String(describing: computeUnits), privacy: .public)")
    let model = try MLModel(contentsOf: url, configuration: configuration)
    let inputs = model.modelDescription.inputDescriptionsByName.keys.sorted().joined(separator: ",")
    let outputs = model.modelDescription.outputDescriptionsByName.keys.sorted().joined(separator: ",")
    logger.info("loaded model name=\(name, privacy: .public) inputs=\(inputs, privacy: .public) outputs=\(outputs, privacy: .public)")
    return model
  }
}

enum ModelError: LocalizedError {
  case missingModel(String)
  case missingOutput(String)
  case invalidOutput(String)

  var errorDescription: String? {
    switch self {
    case .missingModel(let name): "缺少模型：\(name)"
    case .missingOutput(let name): "模型缺少输出：\(name)"
    case .invalidOutput(let name): "模型输出格式错误：\(name)"
    }
  }
}
