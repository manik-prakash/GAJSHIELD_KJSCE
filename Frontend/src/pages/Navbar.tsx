import React from "react";
import { NavLink } from "react-router-dom";
import { Shield } from "lucide-react";
import { Button } from "../components/ui/button";

const Navbar: React.FC = () => {
  return (
    <header className="sticky top-0 z-50 w-full bg-black backdrop-blur pr-5 pl-5">
      <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0">
        <NavLink to="/" className="flex gap-2 items-center text-xl text-white font-bold">
          <Shield className="h-6 w-6 text-white" />
          <span>Dr. Malware</span>
        </NavLink>
        <div className="flex flex-1 items-center justify-end space-x-4">
          <nav className="flex items-center space-x-2">
            <Button className="bg-[#ffffff] hover:bg-[#000000] hover:text-white">
              <NavLink to="/dashboard" className={({ isActive }) => isActive ? "underline text-primary" : ""}>
                Dashboard
              </NavLink>
            </Button>
            <Button className="bg-[#ffffff] hover:bg-[#000000] hover:text-white">
              <NavLink to="/history" className={({ isActive }) => isActive ? "underline text-primary" : ""}>
                History
              </NavLink>
            </Button>
            <Button className="bg-[#ffffff] hover:bg-[#000000] hover:text-white">
              <NavLink to="/settings" className={({ isActive }) => isActive ? "underline text-primary" : ""}>
                Settings
              </NavLink>
            </Button>
            <Button className="bg-[#ffffff] hover:bg-[#000000] hover:text-white">
              <NavLink to="/scan" className={({ isActive }) => isActive ? "underline text-primary" : "" }>
                New Scan
              </NavLink>
            </Button>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
