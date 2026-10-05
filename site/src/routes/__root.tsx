import { createRootRoute, HeadContent, Outlet, Scripts } from "@tanstack/react-router";

import { SiteProvider } from "@/components/site-provider";
import { DEFAULT_LOCALE } from "@/lib/i18n";
import appCss from "@/styles/app.css?url";

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      {
        title: "RVC Next",
      },
      {
        name: "description",
        content: "RVC Next：转换、实时变声、人声分离、训练与模型管理的一体化 RVC。",
      },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
      {
        rel: "icon",
        type: "image/svg+xml",
        href: `${import.meta.env.BASE_URL}favicon.svg`,
      },
    ],
    scripts: [
      {
        src: "https://licyk-umami.netlify.app/script.js",
        async: true,
        "data-website-id": "308fc79d-d064-456b-9e02-5d45b944e030",
      },
    ],
  }),
  component: RootComponent,
});

function RootComponent() {
  return (
    <html lang={DEFAULT_LOCALE} suppressHydrationWarning>
      <head>
        <HeadContent />
      </head>
      <body className="flex min-h-screen flex-col">
        <SiteProvider>
          <Outlet />
        </SiteProvider>
        <Scripts />
      </body>
    </html>
  );
}
