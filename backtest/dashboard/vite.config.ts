import { defineConfig } from "vite";

// Served under /dashboard/ so `npm run dev` opens http://localhost:5173/dashboard/.
export default defineConfig({
  base: "/dashboard/",
  server: { open: "/dashboard/" },
  preview: { open: "/dashboard/" },
});
