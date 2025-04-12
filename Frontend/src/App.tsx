import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Scan from "./pages/scan";
import Auth from "./pages/Auth";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/dashboard",
    element: <Dashboard />,
  },
  {
    path: "/history",
    element: <History />,
  },
  {
    path: "/scan",
    element: <Scan />,
  },
  {
    path: "/auth",
    element: <Auth />,
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
