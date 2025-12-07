import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("analysis", "routes/analysis.tsx"),
  route("correlations", "routes/correlations.tsx"),
  route("trends", "routes/trends.tsx"),
  route("comparison", "routes/comparison.tsx"),
  route("loan-needs", "routes/loan-needs.tsx"),
  route("methodology", "routes/methodology.tsx"),
] satisfies RouteConfig;
