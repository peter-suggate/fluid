"use client";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ThemeSwitch } from "../components/ThemeSwitch";
import { labMethodFromSearch, type LabMethod as Method } from "./lab-method";
import { replaceLocationSearch } from "../lib/core/query-state-sync";
import css from "./Lab.module.css";
const Uniform = dynamic(
  () => import("./UniformLab").then((m) => m.UniformLab),
  { ssr: false, loading: () => <p>Loading Uniform Geometric…</p> },
);
const Adaptive = dynamic(
  () => import("./AdvanceLab").then((m) => m.AdvanceLab),
  { ssr: false, loading: () => <p>Loading Adaptive Geometric…</p> },
);
export function Lab() {
  const [method, setMethod] = useState<Method>();
  useEffect(() => {
    const sync = () => setMethod(labMethodFromSearch(location.search));
    sync();
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, []);
  const select = (next: Method) => {
    const url = new URL(location.href);
    url.searchParams.set("method", next);
    replaceLocationSearch(url.searchParams.toString());
    setMethod(next);
  };
  return (
    <div className={css.lab}>
      <header className={css.header}>
        <Link href="/">Fluid Lab</Link>
        <label>
          2D advance{" "}
          <select
            aria-label="Method"
            value={method ?? "uniform-volume"}
            onChange={(e) => select(e.target.value as Method)}
          >
            <option value="uniform-volume">Uniform Geometric</option>
            <option value="adaptive-volume">Adaptive Geometric</option>
          </select>
        </label>
        <ThemeSwitch />
      </header>
      <div className={css.body}>
        {method === "uniform-volume" ? (
          <Uniform />
        ) : method === "adaptive-volume" ? (
          <Adaptive />
        ) : (
          <p>Loading lab…</p>
        )}
      </div>
    </div>
  );
}
