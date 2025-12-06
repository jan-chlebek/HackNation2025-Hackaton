import { clsx } from "clsx";

interface DashboardCardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}

export function DashboardCard({ title, children, className, action }: DashboardCardProps) {
  return (
    <div className={clsx("pko-card p-6 flex flex-col", className)}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-pko-navy">{title}</h3>
        {action && <div>{action}</div>}
      </div>
      <div className="flex-1">
        {children}
      </div>
    </div>
  );
}
