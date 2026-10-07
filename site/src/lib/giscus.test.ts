import { describe, expect, it } from "vitest";

import { GISCUS_ATTRIBUTES, giscusLanguage, giscusTerm, giscusTheme } from "@/lib/giscus";
import { LOCALES } from "@/lib/i18n";

describe("giscus", () => {
  it("uses rvc-next's discussions, mapped by the page's own term", () => {
    expect(GISCUS_ATTRIBUTES["data-repo"]).toBe("licyk/rvc-next");
    expect(GISCUS_ATTRIBUTES["data-category"]).toBe("Comment");
    expect(GISCUS_ATTRIBUTES["data-mapping"]).toBe("specific");
    expect(GISCUS_ATTRIBUTES["data-strict"]).toBe("1");
  });

  it("maps one discussion to each page, with or without a trailing slash", () => {
    expect(giscusTerm("/docs/quick-start", "/")).toBe("docs/quick-start");
    expect(giscusTerm("/docs/quick-start/", "/")).toBe("docs/quick-start");
    expect(giscusTerm("/docs/", "/")).toBe("docs");
    expect(giscusTerm("/", "/")).toBe("index");
  });

  it("gives a page the same discussion on GitHub Pages (/rvc-next/) and at a root", () => {
    expect(giscusTerm("/rvc-next/docs/guide-live", "/rvc-next/")).toBe("docs/guide-live");
    expect(giscusTerm("/rvc-next/docs/guide-live/", "/rvc-next")).toBe("docs/guide-live");
    expect(giscusTerm("/rvc-next/", "/rvc-next/")).toBe("index");
    expect(giscusTerm("/rvc-nextra/docs", "/rvc-next/")).toBe("rvc-nextra/docs");
  });

  it("gives both languages of a page the same discussion", () => {
    expect(giscusTerm("/en/docs/guide-live", "/")).toBe("docs/guide-live");
    expect(giscusTerm("/rvc-next/zh-CN/docs/guide-live", "/rvc-next/")).toBe("docs/guide-live");
  });

  it("follows the site's resolved theme", () => {
    expect(giscusTheme("dark")).toBe("dark");
    expect(giscusTheme("light")).toBe("light");
    expect(giscusTheme(undefined)).toBe("light");
  });

  it("has a giscus language for every site locale", () => {
    for (const locale of LOCALES) expect(giscusLanguage(locale)).toBeTruthy();
    expect(giscusLanguage("zh-CN")).toBe("zh-CN");
    expect(giscusLanguage("en")).toBe("en");
  });
});
