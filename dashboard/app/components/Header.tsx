import { Link } from "react-router";
import { Github } from "lucide-react";

export function Header() {
  return (
    <header className="bg-white border-b border-pko-gray-medium/30 h-16 flex items-center px-6 sticky top-0 z-50">
      <div className="flex items-center gap-4">
        <div className="w-8 h-8 bg-pko-navy rounded-sm flex items-center justify-center text-white font-bold text-xs">
          PKO
        </div>
        <h1 className="text-xl font-bold text-pko-navy tracking-tight">
          Analizy Sektorowe
        </h1>
      </div>
      <div className="ml-auto flex items-center gap-4">
        <div className="text-sm text-pko-navy/70">
          HackNation 2025
        </div>
        <a 
          href="https://github.com/jan-chlebek/HackNation2025-Hackaton" 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-pko-navy/70 hover:text-pko-navy transition-colors"
        >
          <Github size={24} />
        </a>
      </div>
    </header>
  );
}
