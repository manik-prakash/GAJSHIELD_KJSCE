"use client";

import { useState } from "react";
import {
  BarChart2,
  PieChart,
  Calendar,
  Clock,
  ArrowUpRight,
  ArrowUpCircle,
  Shield,
  AlertCircle,
  Zap,
  ChevronRight,
  RefreshCcw,
  Filter
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import Navbar from "./Navbar";
import Footer from "./Footer";
import { Progress } from "@/components/ui/progress";

export default function DashboardPage() {
  const [dateRange, setDateRange] = useState("7d");
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-slate-50 dark:from-background dark:to-slate-950">
      <Navbar />
      <div className="container px-4 py-6 mx-auto max-w-7xl">
        <main className="flex-1">
          <div className="flex flex-col gap-8">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <Shield className="h-8 w-8 text-primary" />
                  <h1 className="text-3xl font-bold">Security Dashboard</h1>
                </div>
                <p className="text-muted-foreground mt-2">
                  Real-time malware detection and security insights
                </p>
              </div>
              <div className="flex items-center gap-3 self-end md:self-auto">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex items-center gap-2"
                  onClick={handleRefresh}
                >
                  <RefreshCcw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
                <Tabs 
                  defaultValue={dateRange} 
                  onValueChange={setDateRange}
                  className="bg-background/60 backdrop-blur-sm rounded-lg border shadow-sm"
                >
                  <TabsList>
                    <TabsTrigger value="24h">24h</TabsTrigger>
                    <TabsTrigger value="7d">7d</TabsTrigger>
                    <TabsTrigger value="30d">30d</TabsTrigger>
                    <TabsTrigger value="all">All</TabsTrigger>
                  </TabsList>
                </Tabs>
                <Button variant="outline" size="icon" className="bg-background/60 backdrop-blur-sm">
                  <Calendar className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Status Summary */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card className="overflow-hidden border-l-4 border-l-primary shadow-md hover:shadow-lg transition-all">
                <CardHeader className="pb-2">
                  <div className="flex justify-between">
                    <CardDescription className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-primary" />
                      Total Scans
                    </CardDescription>
                    <div className="p-2 bg-primary/10 rounded-full">
                      <ArrowUpCircle className="h-5 w-5 text-primary" />
                    </div>
                  </div>
                  <CardTitle className="text-4xl font-bold">1,248</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs font-medium flex items-center">
                    <ArrowUpRight className="h-3 w-3 mr-1 text-primary" />
                    <span className="text-primary">12%</span>
                    <span className="ml-1 text-muted-foreground">from last period</span>
                  </div>
                  <Progress value={72} className="h-1 mt-3" />
                </CardContent>
              </Card>

              <Card className="overflow-hidden border-l-4 border-l-red-500 shadow-md hover:shadow-lg transition-all">
                <CardHeader className="pb-2">
                  <div className="flex justify-between">
                    <CardDescription className="flex items-center gap-2">
                      <AlertCircle className="h-4 w-4 text-red-500" />
                      Malware Detected
                    </CardDescription>
                    <div className="p-2 bg-red-500/10 rounded-full">
                      <Shield className="h-5 w-5 text-red-500" />
                    </div>
                  </div>
                  <CardTitle className="text-4xl font-bold">37</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs font-medium flex items-center">
                    <ArrowUpRight className="h-3 w-3 mr-1 text-red-500" />
                    <span className="text-red-500">8%</span>
                    <span className="ml-1 text-muted-foreground">from last period</span>
                  </div>
                  <Progress value={28} className="h-1 mt-3 bg-red-100 dark:bg-red-950">
                    <div className="bg-red-500 h-full w-[28%] rounded-full" />
                  </Progress>
                </CardContent>
              </Card>

              <Card className="overflow-hidden border-l-4 border-l-blue-500 shadow-md hover:shadow-lg transition-all">
                <CardHeader className="pb-2">
                  <div className="flex justify-between">
                    <CardDescription className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-blue-500" />
                      Files Scanned
                    </CardDescription>
                    <div className="p-2 bg-blue-500/10 rounded-full">
                      <BarChart2 className="h-5 w-5 text-blue-500" />
                    </div>
                  </div>
                  <CardTitle className="text-4xl font-bold">5,392</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs font-medium flex items-center">
                    <ArrowUpRight className="h-3 w-3 mr-1 text-blue-500" />
                    <span className="text-blue-500">24%</span>
                    <span className="ml-1 text-muted-foreground">from last period</span>
                  </div>
                  <Progress value={86} className="h-1 mt-3 bg-blue-100 dark:bg-blue-950">
                    <div className="bg-blue-500 h-full w-[86%] rounded-full" />
                  </Progress>
                </CardContent>
              </Card>

              <Card className="overflow-hidden border-l-4 border-l-green-500 shadow-md hover:shadow-lg transition-all">
                <CardHeader className="pb-2">
                  <div className="flex justify-between">
                    <CardDescription className="flex items-center gap-2">
                      <Shield className="h-4 w-4 text-green-500" />
                      Detection Rate
                    </CardDescription>
                    <div className="p-2 bg-green-500/10 rounded-full">
                      <PieChart className="h-5 w-5 text-green-500" />
                    </div>
                  </div>
                  <CardTitle className="text-4xl font-bold">99.2%</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs font-medium flex items-center">
                    <ArrowUpRight className="h-3 w-3 mr-1 text-green-500" />
                    <span className="text-green-500">1.5%</span>
                    <span className="ml-1 text-muted-foreground">from last period</span>
                  </div>
                  <Progress value={99} className="h-1 mt-3 bg-green-100 dark:bg-green-950">
                    <div className="bg-green-500 h-full w-[99%] rounded-full" />
                  </Progress>
                </CardContent>
              </Card>
            </div>

            {/* Charts Section */}
            <div className="grid gap-6 lg:grid-cols-2">
              <Card className="shadow-md hover:shadow-lg transition-all">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <PieChart className="h-5 w-5 text-primary" />
                        Threat Distribution
                      </CardTitle>
                      <CardDescription>
                        Types of malware detected in your scans
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon">
                      <Filter className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="h-80 flex items-center justify-center bg-slate-50 dark:bg-slate-900/50 rounded-lg">
                    <PieChart className="h-64 w-64 text-muted-foreground" />
                  </div>
                </CardContent>
                <CardFooter className="border-t pt-4">
                  <div className="grid grid-cols-2 gap-4 w-full text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-500"></div>
                      <span className="font-medium">Trojans</span>
                      <Badge variant="outline" className="ml-auto">42%</Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                      <span className="font-medium">Ransomware</span>
                      <Badge variant="outline" className="ml-auto">18%</Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                      <span className="font-medium">Spyware</span>
                      <Badge variant="outline" className="ml-auto">24%</Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-green-500"></div>
                      <span className="font-medium">Other</span>
                      <Badge variant="outline" className="ml-auto">16%</Badge>
                    </div>
                  </div>
                </CardFooter>
              </Card>

              <Card className="shadow-md hover:shadow-lg transition-all">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <BarChart2 className="h-5 w-5 text-primary" />
                        Scan Activity
                      </CardTitle>
                      <CardDescription>Number of scans over time</CardDescription>
                    </div>
                    <Button variant="ghost" size="icon">
                      <Filter className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="h-80 flex items-center justify-center bg-slate-50 dark:bg-slate-900/50 rounded-lg">
                    <BarChart2 className="h-64 w-full text-muted-foreground" />
                  </div>
                </CardContent>
                <CardFooter className="border-t pt-4">
                  <div className="grid grid-cols-2 gap-4 w-full">
                    <div className="bg-slate-100 dark:bg-slate-900 p-3 rounded-lg">
                      <div className="text-sm text-muted-foreground">Total Scans</div>
                      <div className="text-xl font-bold">1,248</div>
                    </div>
                    <div className="bg-slate-100 dark:bg-slate-900 p-3 rounded-lg">
                      <div className="text-sm text-muted-foreground">Avg. per day</div>
                      <div className="text-xl font-bold">178</div>
                    </div>
                  </div>
                </CardFooter>
              </Card>
            </div>

            {/* Recent Detections Table */}
            <Card className="shadow-md hover:shadow-xl transition-all">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <AlertCircle className="h-5 w-5 text-red-500" />
                      Recent Detections
                    </CardTitle>
                    <CardDescription>
                      Latest malware and suspicious files detected
                    </CardDescription>
                  </div>
                  <Button variant="outline" size="sm" className="flex items-center gap-2">
                    <Filter className="h-4 w-4" />
                    Filter
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="rounded-xl overflow-hidden border shadow-sm">
                  <div className="grid grid-cols-12 gap-4 p-4 bg-primary/5 font-medium text-sm">
                    <div className="col-span-4">Filename</div>
                    <div className="col-span-3">Threat Type</div>
                    <div className="col-span-2">Status</div>
                    <div className="col-span-2">Detected</div>
                    <div className="col-span-1">Actions</div>
                  </div>

                  <div className="grid grid-cols-12 gap-4 p-4 border-t items-center text-sm hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors">
                    <div className="col-span-4 font-medium flex items-center gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      setup_installer.exe
                    </div>
                    <div className="col-span-3 text-muted-foreground">
                      Trojan.Win32.Miner
                    </div>
                    <div className="col-span-2">
                      <Badge className="bg-red-500 hover:bg-red-600">
                        Malware
                      </Badge>
                    </div>
                    <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                      <Clock className="h-3 w-3" /> 2 hours ago
                    </div>
                    <div className="col-span-1">
                      <Button variant="ghost" size="sm" className="rounded-full">
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="grid grid-cols-12 gap-4 p-4 border-t items-center text-sm hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors">
                    <div className="col-span-4 font-medium flex items-center gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      invoice_april.docx
                    </div>
                    <div className="col-span-3 text-muted-foreground">
                      Macro.Downloader
                    </div>
                    <div className="col-span-2">
                      <Badge className="bg-red-500 hover:bg-red-600">
                        Malware
                      </Badge>
                    </div>
                    <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                      <Clock className="h-3 w-3" /> 5 hours ago
                    </div>
                    <div className="col-span-1">
                      <Button variant="ghost" size="sm" className="rounded-full">
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="grid grid-cols-12 gap-4 p-4 border-t items-center text-sm hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors">
                    <div className="col-span-4 font-medium flex items-center gap-2">
                      <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                      system_update.bat
                    </div>
                    <div className="col-span-3 text-muted-foreground">
                      Suspicious Script
                    </div>
                    <div className="col-span-2">
                      <Badge className="bg-yellow-500 hover:bg-yellow-600">
                        Suspicious
                      </Badge>
                    </div>
                    <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                      <Clock className="h-3 w-3" /> 1 day ago
                    </div>
                    <div className="col-span-1">
                      <Button variant="ghost" size="sm" className="rounded-full">
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="grid grid-cols-12 gap-4 p-4 border-t items-center text-sm hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors">
                    <div className="col-span-4 font-medium flex items-center gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      crypto_wallet.exe
                    </div>
                    <div className="col-span-3 text-muted-foreground">
                      Ransomware.Cryptolocker
                    </div>
                    <div className="col-span-2">
                      <Badge className="bg-red-500 hover:bg-red-600">
                        Malware
                      </Badge>
                    </div>
                    <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                      <Clock className="h-3 w-3" /> 2 days ago
                    </div>
                    <div className="col-span-1">
                      <Button variant="ghost" size="sm" className="rounded-full">
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="grid grid-cols-12 gap-4 p-4 border-t items-center text-sm hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors">
                    <div className="col-span-4 font-medium flex items-center gap-2">
                      <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                      backup_tool.exe
                    </div>
                    <div className="col-span-3 text-muted-foreground">
                      PUA.Adware
                    </div>
                    <div className="col-span-2">
                      <Badge className="bg-yellow-500 hover:bg-yellow-600">
                        Suspicious
                      </Badge>
                    </div>
                    <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                      <Clock className="h-3 w-3" /> 3 days ago
                    </div>
                    <div className="col-span-1">
                      <Button variant="ghost" size="sm" className="rounded-full">
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="border-t pt-4 flex justify-center">
                <Button className="bg-primary hover:bg-primary/90 text-white flex items-center gap-2">
                  View All Detections
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </CardFooter>
            </Card>
          </div>
        </main>
      </div>
      <Footer />
    </div>
  );
}