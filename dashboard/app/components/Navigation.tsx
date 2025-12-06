import { Link, useLocation } from "react-router";

const PKO_COLORS = {
  navy: "#1a2f3a",
  red: "#d93026",
  gold: "#c9a961",
  grayLight: "#f5f5f5",
};

export function Navigation() {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  const navItems = [
    { path: "/", label: "Home", icon: "🏠" },
    { path: "/analytics", label: "Multi-Method Analysis", icon: "📊" },
    { path: "/financial-indicators", label: "Financial Indicators", icon: "📈" },
  ];

  return (
    <nav className="bg-white shadow-sm border-b" style={{ borderColor: PKO_COLORS.grayLight }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex space-x-8">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`inline-flex items-center px-1 pt-1 pb-4 border-b-2 text-sm font-medium transition-colors ${
                isActive(item.path)
                  ? "border-current"
                  : "border-transparent hover:border-gray-300"
              }`}
              style={{
                color: isActive(item.path) ? PKO_COLORS.navy : "#6b7280",
                borderBottomColor: isActive(item.path) ? PKO_COLORS.red : "transparent",
              }}
            >
              <span className="mr-2">{item.icon}</span>
              {item.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
