import React from "react";
import { createRoot } from "react-dom/client";
import { IsaacLens } from "../app/IsaacLens";
import "../app/globals.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <IsaacLens />
  </React.StrictMode>,
);
