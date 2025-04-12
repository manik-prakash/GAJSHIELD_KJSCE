import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Scan from "./pages/Scan";
import Auth from "./pages/Auth";
import Chatbox from "./pages/Chatbox";

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
  {
    path: "/chatbox",
    element: <Chatbox />,
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
