import Foundation
import UIKit

struct VectorMetadata: Decodable {
  let rows: Int
  let dimensions: Int
  let objects: [VectorObject]
}

struct VectorObject: Decodable {
  let vectorIndex: Int
  let id: String
}

final class ItemRepository: @unchecked Sendable {
  let objects: [IsaacObject]
  let vectors: [Float]
  let dimensions: Int
  private let byID: [String: IsaacObject]

  init(bundle: Bundle = .main) throws {
    let objectURL = try Self.resourceURL("objects", "json", bundle: bundle)
    let vectorURL = try Self.resourceURL("item-vectors", "f16", bundle: bundle)
    let metadataURL = try Self.resourceURL("item-vectors", "json", bundle: bundle)
    let decoded = try JSONDecoder().decode([IsaacObject].self, from: Data(contentsOf: objectURL))
    let metadata = try JSONDecoder().decode(VectorMetadata.self, from: Data(contentsOf: metadataURL))
    let raw = try Data(contentsOf: vectorURL)
    let expectedBytes = metadata.rows * metadata.dimensions * MemoryLayout<UInt16>.size
    guard raw.count == expectedBytes else {
      throw RepositoryError.invalidVectorSize(expected: expectedBytes, actual: raw.count)
    }
    var decodedVectors = [Float]()
    decodedVectors.reserveCapacity(metadata.rows * metadata.dimensions)
    raw.withUnsafeBytes { buffer in
      let words = buffer.bindMemory(to: UInt16.self)
      for word in words {
        decodedVectors.append(Float(Float16(bitPattern: UInt16(littleEndian: word))))
      }
    }
    let objectMap = Dictionary(uniqueKeysWithValues: decoded.map { ($0.id, $0) })
    guard metadata.objects.allSatisfy({ objectMap[$0.id] != nil }) else {
      throw RepositoryError.metadataMismatch
    }
    objects = metadata.objects.compactMap { objectMap[$0.id] }
    vectors = decodedVectors
    dimensions = metadata.dimensions
    byID = objectMap
  }

  func item(id: String) -> IsaacObject? {
    byID[id]
  }

  func icon(for item: IsaacObject) -> UIImage? {
    let path = item.iconPath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    let file = URL(fileURLWithPath: path)
    let resource = file.deletingPathExtension().lastPathComponent
    let subdirectory = file.deletingLastPathComponent().path
    guard let url = Bundle.main.url(forResource: resource, withExtension: "png", subdirectory: subdirectory) else {
      return nil
    }
    return UIImage(contentsOfFile: url.path)
  }

  private static func resourceURL(_ name: String, _ extensionName: String, bundle: Bundle) throws -> URL {
    guard let url = bundle.url(forResource: name, withExtension: extensionName) else {
      throw RepositoryError.missingResource("\(name).\(extensionName)")
    }
    return url
  }
}

enum RepositoryError: LocalizedError {
  case missingResource(String)
  case invalidVectorSize(expected: Int, actual: Int)
  case metadataMismatch

  var errorDescription: String? {
    switch self {
    case .missingResource(let name): "缺少离线资源：\(name)"
    case .invalidVectorSize(let expected, let actual): "向量数据大小错误：应为 \(expected)，实际为 \(actual)"
    case .metadataMismatch: "百科与向量索引不匹配"
    }
  }
}
