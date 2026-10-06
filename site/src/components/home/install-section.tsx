import Link from "fumadocs-core/link";
import { ArrowRight } from "lucide-react";

import HomeFooter from "@/components/home/home-footer";
import PageReveal from "@/components/page/page-reveal";
import PageSection from "@/components/page/page-section";
import { useSiteCopy } from "@/lib/site-copy";

/** The README's install path; the guide covers other platforms and hardware. */
const COMMANDS: readonly (readonly [command: string, comment: string])[] = [
  ["python -m venv .venv && . .venv/bin/activate", ""],
  ["pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130", "# PyTorch"],
  ["pip install rvc-next", ""],
  ["rvc-next assets download", "# HuBERT, RMVPE, FCPE"],
  ["rvc-next webui", "# http://127.0.0.1:7868"],
];

export default function InstallSection() {
  const copy = useSiteCopy().home.install;

  return (
    <PageSection className="is-outro" id="install" labelledBy="install-title">
      <PageReveal>
        <div className="home-install-card page-interactive">
          <div className="home-install-copy">
            <span>INSTALL</span>
            <h2 id="install-title">{copy.title}</h2>
            <p>{copy.description}</p>
            <pre className="home-install-commands">
              {COMMANDS.map(([command, comment]) => (
                <span key={command}>
                  {command}
                  {comment ? <span className="comment">{`  ${comment}`}</span> : null}
                  {"\n"}
                </span>
              ))}
            </pre>
          </div>
          <Link className="home-install-link page-interactive" href="/docs/installation">
            {copy.link}
            <ArrowRight className="page-interactive-part" aria-hidden="true" />
          </Link>
        </div>
      </PageReveal>
      <HomeFooter />
    </PageSection>
  );
}
