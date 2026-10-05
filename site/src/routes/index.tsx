import { createFileRoute } from "@tanstack/react-router";
import { HomeLayout } from "fumadocs-ui/layouts/home";
import { useRef } from "react";

import FeaturesSection from "@/components/home/features-section";
import HeroSection from "@/components/home/hero-section";
import HighlightsSection from "@/components/home/highlights-section";
import InstallSection from "@/components/home/install-section";
import WorkflowSection from "@/components/home/workflow-section";
import SectionRail from "@/components/page/section-rail";
import { useDocumentMetadata } from "@/lib/document-metadata";
import { HOME_SECTION_IDS, homeSections } from "@/lib/home-sections";
import type { Locale } from "@/lib/i18n";
import { baseOptions } from "@/lib/layout.shared";
import { useActiveSection, useScrollReveal } from "@/lib/page-motion";
import { useSiteLocale } from "@/lib/site-copy";

const METADATA: Record<Locale, { description: string; title: string }> = {
  "zh-CN": {
    title: "RVC Next — 基于检索的语音转换，一体化重写",
    description:
      "转换、实时变声、人声分离、训练与模型管理，由同一个服务端、网页界面和命令行提供，输出与原版 RVC 一致。",
  },
  en: {
    title: "RVC Next — Retrieval-based Voice Conversion, rewritten as one app",
    description:
      "Conversion, live voice changing, vocal separation, training and model management from one server, web UI and command line, with the original RVC's output.",
  },
};

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: METADATA["zh-CN"].title },
      { name: "description", content: METADATA["zh-CN"].description },
    ],
  }),
  component: Home,
});

function Home() {
  const locale = useSiteLocale();
  const homeRef = useRef<HTMLElement>(null);
  const sections = homeSections(locale);
  const activeSectionId = useActiveSection(HOME_SECTION_IDS);

  useDocumentMetadata(METADATA[locale]);
  useScrollReveal(homeRef);

  return (
    <HomeLayout {...baseOptions(locale)} className="home-page paged" ref={homeRef}>
      <SectionRail activeId={activeSectionId} sections={sections} />
      <HeroSection />
      <FeaturesSection />
      <WorkflowSection />
      <HighlightsSection />
      <InstallSection />
    </HomeLayout>
  );
}
