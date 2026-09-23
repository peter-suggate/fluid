"use client";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ThemeSwitch } from "../components/ThemeSwitch";
import { Select } from "../components/ui";
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
const METHOD_OPTIONS: readonly { readonly value: Method; readonly label: string }[] = [
  { value: "uniform-volume", label: "Uniform Geometric" },
  { value: "adaptive-volume", label: "Adaptive Geometric" },
];
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
          <Select
            ariaLabel="Method"
            value={method ?? "uniform-volume"}
            options={METHOD_OPTIONS}
            onChange={select}
          />
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
