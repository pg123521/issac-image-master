import CoreGraphics
import Foundation
import UIKit

struct IsaacObject: Codable, Identifiable, Hashable {
  let id: String
  let kind: String
  let gameId: Int
  let nameZh: String
  let nameEn: String
  let pickup: String
  let description: String
  let effects: [String]
  let type: String
  let pools: [String]
  let tags: [String]
  let iconPath: String
  let sourceName: String
  let sourceUrl: String
}

struct DetectionRegion: Identifiable, Equatable {
  let id: UUID
  var rect: CGRect

  init(id: UUID = UUID(), rect: CGRect) {
    self.id = id
    self.rect = rect
  }
}

struct SearchMatch: Identifiable {
  let item: IsaacObject
  let score: Float
  var id: String { item.id }
}
