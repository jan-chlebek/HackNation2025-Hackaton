import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("analytics", "routes/analytics.tsx"),
  route("financial-indicators", "routes/financial-indicators.tsx"),
] satisfies RouteConfig;
