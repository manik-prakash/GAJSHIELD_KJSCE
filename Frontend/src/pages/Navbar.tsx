import React, { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { Shield } from "lucide-react";

const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const [dropdownVisible, setDropdownVisible] = useState<boolean>(false);

  const handleLogout = async () => {
    await fetch("http://localhost:8080/logout", {
      method: "POST",
      credentials: "include",
    });
    navigate("/auth");
  };

  return (
    <header className="sticky top-0 z-50 w-full bg-black backdrop-blur px-5">
      <div className="container flex h-16 items-center justify-between">
        <NavLink
          to="/"
          className="flex gap-2 items-center text-xl text-white font-bold hover:opacity-80 transition-opacity"
        >
          <Shield className="h-6 w-6 text-white" />
          <span>Dr. Malware</span>
        </NavLink>

        <div className="flex items-center gap-4">
          <nav className="flex items-center gap-2">
            <NavLink
              to="/dashboard"
              className={({ isActive }) =>
                `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-white text-black"
                    : "text-white hover:bg-gray-800"
                }`
              }
            >
              Dashboard
            </NavLink>
            <NavLink
              to="/scan"
              className={({ isActive }) =>
                `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-white text-black"
                    : "text-white hover:bg-gray-800"
                }`
              }
            >
              New Scan
            </NavLink>

            <NavLink
              to="/chatbox"
              className={({ isActive }) =>
                `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-white text-black"
                    : "text-white hover:bg-gray-800"
                }`
              }
            >
              Chatbot
            </NavLink>

            {document.cookie ? (
                <div className="relative">
                  <button
                  className="px-4 py-2 rounded-md text-sm font-medium text-white hover:bg-white hover:text-black transition-colors"
                  onClick={() => setDropdownVisible((prev) => !prev)}
                  >
                  Welcome, {document.cookie.split("=")[1]}!
                  </button>
                  {dropdownVisible && (
                  <div className="absolute right-2 mt-2 w-40 bg-white rounded-md shadow-lg z-10">
                    <button
                    onClick={handleLogout}
                    className="block w-full px-4 py-2 text-left text-sm text-red-600 "
                    >
                    Logout
                    </button>
                  </div>
                  )}
                </div>
            ) : (
              <NavLink
                to="/auth"
                className={({ isActive }) =>
                  `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-white text-black"
                      : "bg-gray-100 text-black hover:bg-gray-300"
                  }`
                }
              >
                Login / Sign Up
              </NavLink>
            )}
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Navbar;