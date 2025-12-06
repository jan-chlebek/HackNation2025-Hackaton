import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("analysis", "routes/analysis.tsx"),
  route("correlations", "routes/correlations.tsx"),
  route("rankings", "routes/rankings.tsx"),
] satisfies RouteConfig;
