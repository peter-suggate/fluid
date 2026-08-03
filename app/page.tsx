import { SceneLibrary } from "@/components/SceneLibrary";
import { redirect } from "next/navigation";

type HomeProps = {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

export default async function Home({ searchParams }: HomeProps) {
  const incoming = await searchParams;
  // Scene links produced before scenes gained their own page named the preset
  // on /. Keep those bookmarks useful; an explicit library link remains here.
  if (incoming.scene !== undefined && incoming.view !== "library") {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(incoming)) {
      for (const entry of Array.isArray(value) ? value : value === undefined ? [] : [value]) {
        query.append(key, entry);
      }
    }
    redirect(`/scene?${query}`);
  }
  return <SceneLibrary />;
}
