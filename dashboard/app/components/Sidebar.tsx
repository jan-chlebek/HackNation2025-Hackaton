import { Link, useLocation } from "react-router";
import { BarChart3, Home, PieChart, TrendingUp, Activity, FileText } from "lucide-react";
import { clsx } from "clsx";

const navItems = [
  { label: "Dashboard", href: "/", icon: Home },
  { label: "Analiza Metod", href: "/analysis", icon: BarChart3 },
  { label: "Trendy", href: "/trends", icon: TrendingUp },
  { label: "Korelacje", href: "/correlations", icon: Activity },
  { label: "Rankingi", href: "/rankings", icon: FileText },
  { label: "Metodologia", href: "/methodology", icon: PieChart },
];

export function Sidebar() {
  const location = useLocation();

  return (
    <aside className="w-64 bg-pko-navy text-white h-[calc(100vh-4rem)] sticky top-16 flex flex-col">
      <nav className="p-4 space-y-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.href;
          const Icon = item.icon;
          
          return (
            <Link
              key={item.href}
              to={item.href}
              className={clsx(
                "flex items-center gap-3 px-4 py-3 rounded-md transition-all duration-200",
                isActive 
                  ? "bg-white/10 text-white font-medium border-l-4 border-pko-red" 
                  : "text-white/70 hover:bg-white/5 hover:text-white"
              )}
            >
              <Icon size={20} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
      
      <div className="mt-auto p-6 border-t border-white/10">
        <div className="text-xs text-white/50 uppercase tracking-wider font-semibold mb-2">
          Chlebki & Friend
        </div>
        <div className="text-sm text-white/80 flex flex-col gap-1">
          <span>Jan Chlebek</span>
          <span>Marta Chlebek</span>
          <span>Patryk Hubicki</span>
        </div>
      </div>
    </aside>
  );
}
