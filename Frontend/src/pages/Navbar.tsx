import React from "react";
import { NavLink } from "react-router-dom";
import { Shield } from "lucide-react";

const Navbar: React.FC = () => {
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
              to="/history"
              className={({ isActive }) =>
                `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive 
                    ? "bg-white text-black" 
                    : "text-white hover:bg-gray-800"
                }`
              }
            >
              History
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

            {/* Login / User section */}
            {document.cookie ? (
              <div className="ml-4 px-1 py-2 text-white font-medium">
                Welcome, {document.cookie.split('=')[1]}!
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