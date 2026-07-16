import SwiftUI

struct ItemDetailView: View {
  let item: IsaacObject
  let repository: ItemRepository?
  let onClose: () -> Void

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(alignment: .leading, spacing: 20) {
          HStack(spacing: 18) {
            ObjectIcon(item: item, repository: repository)
              .frame(width: 84, height: 84)
            VStack(alignment: .leading, spacing: 5) {
              Text(item.nameZh)
                .font(.title2.bold())
              Text(item.nameEn)
                .foregroundStyle(.secondary)
              Text("#\(item.gameId)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            }
          }
          if !item.type.isEmpty || !item.pools.isEmpty {
            FlowTags(values: ([item.type] + item.pools).filter { !$0.isEmpty })
          }
          VStack(alignment: .leading, spacing: 8) {
            Text("道具说明").font(.headline)
            Text(item.pickup.isEmpty ? item.description : item.pickup)
              .font(.body)
          }
          if !item.effects.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
              Text("效果").font(.headline)
              ForEach(item.effects.prefix(10), id: \.self) { effect in
                Label(effect, systemImage: "diamond.fill")
                  .font(.body)
                  .labelStyle(EffectLabelStyle())
              }
            }
          }
          Text("来源：\(item.sourceName)")
            .font(.footnote)
            .foregroundStyle(.secondary)
        }
        .padding(20)
        .frame(maxWidth: 560, alignment: .leading)
      }
      .navigationTitle("道具详情")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarTrailing) {
          Button(action: onClose) {
            Image(systemName: "xmark.circle.fill")
          }
          .accessibilityLabel("关闭道具详情")
        }
      }
    }
    .presentationDetents([.medium, .large])
    .presentationDragIndicator(.visible)
  }
}

private struct FlowTags: View {
  let values: [String]

  var body: some View {
    ScrollView(.horizontal, showsIndicators: false) {
      HStack(spacing: 8) {
        ForEach(values.prefix(8), id: \.self) { value in
          Text(value)
            .font(.caption)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color.cyan.opacity(0.14), in: Capsule())
        }
      }
    }
  }
}

private struct EffectLabelStyle: LabelStyle {
  func makeBody(configuration: Configuration) -> some View {
    HStack(alignment: .firstTextBaseline, spacing: 9) {
      configuration.icon
        .font(.system(size: 7))
        .foregroundStyle(.yellow)
      configuration.title
    }
  }
}
