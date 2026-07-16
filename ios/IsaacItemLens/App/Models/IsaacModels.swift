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
  let detectorScore: Float
  let automatic: Bool

  init(id: UUID = UUID(), rect: CGRect, detectorScore: Float = 1, automatic: Bool) {
    self.id = id
    self.rect = rect
    self.detectorScore = detectorScore
    self.automatic = automatic
  }
}

struct SearchMatch: Identifiable {
  let item: IsaacObject
  let score: Float
  var id: String { item.id }
}

enum DetectionStage: Equatable {
  case idle
  case detecting
  case verifying
  case complete(Int)
  case failed(String)

  var isPresented: Bool {
    self != .idle
  }
}
