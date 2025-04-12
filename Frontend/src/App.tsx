import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Home from "./Home";
import Dashboard from "./pages/dashboard";
import History from "./History";
import Settings from "./Settings";
import Scan from "./pages/scan";

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
    path: "/settings",
    element: <Settings />,
  },
  {
    path: "/scan",
    element: <Scan />,
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
