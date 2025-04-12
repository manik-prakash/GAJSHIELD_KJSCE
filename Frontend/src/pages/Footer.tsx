import React from "react";
import { Link } from "react-router-dom";
import { Shield } from "lucide-react";
const Footer: React.FC = () => {
    return (
        <footer className="w-full border-t py-6 text-white pr-5 pl-5 bg-black">
            <div className="container flex flex-col items-center justify-between gap-4 md:flex-row">
                <div className="flex gap-2 items-center text-lg font-semibold">
                    <Shield className="h-5 w-5 text-accent" />
                    <span>Dr. Malware</span>
                </div>
                <p className="text-center text-sm text-white/70">
                    © 2024 Dr. Malware AI. All rights reserved.
                </p>
                <div className="flex gap-4">
                    <Link to="#" className="text-sm text-white/70  underline-offset-4 hover:underline">
                        Terms
                    </Link>
                    <Link to="#" className="text-sm text-white/70  underline-offset-4 hover:underline">
                        Privacy
                    </Link>
                </div>
            </div>
        </footer>
    );
};

export default Footer;