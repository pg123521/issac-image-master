import type { Metadata } from "next";
import { IsaacLens } from "./IsaacLens";

export const metadata: Metadata = {
  title: "Isaac Item Lens",
  description: "在 iPhone 浏览器中离线识别截图里的物品候选。",
};

export default function Home() {
  return <IsaacLens />;
}
