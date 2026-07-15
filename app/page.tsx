import type { Metadata } from "next";
import { IsaacLens } from "./IsaacLens";

export const metadata: Metadata = {
  title: "Isaac Item Lens",
  description: "离线识别《以撒的结合》截图中的道具候选并查询本地百科。",
};

export default function Home() {
  return <IsaacLens />;
}
