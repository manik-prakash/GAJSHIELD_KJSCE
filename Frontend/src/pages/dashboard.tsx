"use client"

import { useState } from "react"
import Link from "next/link"
import { Shield, BarChart2, PieChart, Calendar, Clock, ArrowUpRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"

export default function DashboardPage() {
  const [dateRange, setDateRange] = useState("7d")

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0">
          <Link href="/" className="flex gap-2 items-center text-xl font-bold">
            <Shield className="h-6 w-6 text-primary" />
            <span>MalwareGuard AI</span>
          </Link>
          <div className="flex flex-1 items-center justify-end space-x-4">
            <nav className="flex items-center space-x-2">
              <Button asChild variant="ghost">
                <Link href="/dashboard">Dashboard</Link>
              </Button>
              <Button asChild variant="ghost">
                <Link href="/history">History</Link>
              </Button>
              <Button asChild variant="ghost">
                <Link href="/settings">Settings</Link>
              </Button>
              <Button asChild>
                <Link href="/scan">New Scan</Link>
              </Button>
            </nav>
          </div>
        </div>
      </header>
      <main className="flex-1 container py-8">
        <div className="flex flex-col gap-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
              <p className="text-muted-foreground">Overview of your malware detection activity</p>
            </div>
            <div className="flex items-center gap-2">
              <Tabs defaultValue={dateRange} onValueChange={setDateRange}>
                <TabsList>
                  <TabsTrigger value="24h">24h</TabsTrigger>
                  <TabsTrigger value="7d">7d</TabsTrigger>
                  <TabsTrigger value="30d">30d</TabsTrigger>
                  <TabsTrigger value="all">All</TabsTrigger>
                </TabsList>
              </Tabs>
              <Button variant="outline" size="icon">
                <Calendar className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Total Scans</CardDescription>
                <CardTitle className="text-3xl">1,248</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground flex items-center">
                  <ArrowUpRight className="h-3 w-3 mr-1 text-primary" />
                  <span className="text-primary font-medium">12%</span>
                  <span className="ml-1">from last period</span>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Malware Detected</CardDescription>
                <CardTitle className="text-3xl">37</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground flex items-center">
                  <ArrowUpRight className="h-3 w-3 mr-1 text-red-500" />
                  <span className="text-red-500 font-medium">8%</span>
                  <span className="ml-1">from last period</span>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Files Scanned</CardDescription>
                <CardTitle className="text-3xl">5,392</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground flex items-center">
                  <ArrowUpRight className="h-3 w-3 mr-1 text-primary" />
                  <span className="text-primary font-medium">24%</span>
                  <span className="ml-1">from last period</span>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Detection Rate</CardDescription>
                <CardTitle className="text-3xl">99.2%</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground flex items-center">
                  <ArrowUpRight className="h-3 w-3 mr-1 text-primary" />
                  <span className="text-primary font-medium">1.5%</span>
                  <span className="ml-1">from last period</span>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle>Threat Distribution</CardTitle>
                <CardDescription>Types of malware detected in your scans</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80 flex items-center justify-center">
                  <PieChart className="h-64 w-64 text-muted-foreground" />
                </div>
              </CardContent>
              <CardFooter className="border-t pt-4">
                <div className="grid grid-cols-2 gap-4 w-full text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <span>Trojans (42%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span>Ransomware (18%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                    <span>Spyware (24%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span>Other (16%)</span>
                  </div>
                </div>
              </CardFooter>
            </Card>

            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle>Scan Activity</CardTitle>
                <CardDescription>Number of scans over time</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80 flex items-center justify-center">
                  <BarChart2 className="h-64 w-full text-muted-foreground" />
                </div>
              </CardContent>
              <CardFooter className="border-t pt-4">
                <div className="flex justify-between w-full text-sm text-muted-foreground">
                  <div>Total Scans: 1,248</div>
                  <div>Avg. per day: 178</div>
                </div>
              </CardFooter>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Recent Detections</CardTitle>
              <CardDescription>Latest malware and suspicious files detected</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="border rounded-lg overflow-hidden">
                <div className="grid grid-cols-12 gap-4 p-3 bg-muted font-medium text-sm">
                  <div className="col-span-4">Filename</div>
                  <div className="col-span-3">Threat Type</div>
                  <div className="col-span-2">Status</div>
                  <div className="col-span-2">Detected</div>
                  <div className="col-span-1">Actions</div>
                </div>

                <div className="grid grid-cols-12 gap-4 p-3 border-t items-center text-sm">
                  <div className="col-span-4 font-medium">setup_installer.exe</div>
                  <div className="col-span-3 text-muted-foreground">Trojan.Win32.Miner</div>
                  <div className="col-span-2">
                    <Badge className="bg-red-500 hover:bg-red-600">Malware</Badge>
                  </div>
                  <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" /> 2 hours ago
                  </div>
                  <div className="col-span-1">
                    <Button variant="ghost" size="sm">
                      Details
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-12 gap-4 p-3 border-t items-center text-sm">
                  <div className="col-span-4 font-medium">invoice_april.docx</div>
                  <div className="col-span-3 text-muted-foreground">Macro.Downloader</div>
                  <div className="col-span-2">
                    <Badge className="bg-red-500 hover:bg-red-600">Malware</Badge>
                  </div>
                  <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" /> 5 hours ago
                  </div>
                  <div className="col-span-1">
                    <Button variant="ghost" size="sm">
                      Details
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-12 gap-4 p-3 border-t items-center text-sm">
                  <div className="col-span-4 font-medium">system_update.bat</div>
                  <div className="col-span-3 text-muted-foreground">Suspicious Script</div>
                  <div className="col-span-2">
                    <Badge className="bg-yellow-500 hover:bg-yellow-600">Suspicious</Badge>
                  </div>
                  <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" /> 1 day ago
                  </div>
                  <div className="col-span-1">
                    <Button variant="ghost" size="sm">
                      Details
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-12 gap-4 p-3 border-t items-center text-sm">
                  <div className="col-span-4 font-medium">crypto_wallet.exe</div>
                  <div className="col-span-3 text-muted-foreground">Ransomware.Cryptolocker</div>
                  <div className="col-span-2">
                    <Badge className="bg-red-500 hover:bg-red-600">Malware</Badge>
                  </div>
                  <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" /> 2 days ago
                  </div>
                  <div className="col-span-1">
                    <Button variant="ghost" size="sm">
                      Details
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-12 gap-4 p-3 border-t items-center text-sm">
                  <div className="col-span-4 font-medium">backup_tool.exe</div>
                  <div className="col-span-3 text-muted-foreground">PUA.Adware</div>
                  <div className="col-span-2">
                    <Badge className="bg-yellow-500 hover:bg-yellow-600">Suspicious</Badge>
                  </div>
                  <div className="col-span-2 text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" /> 3 days ago
                  </div>
                  <div className="col-span-1">
                    <Button variant="ghost" size="sm">
                      Details
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="border-t pt-4 flex justify-center">
              <Button variant="outline">View All Detections</Button>
            </CardFooter>
          </Card>
        </div>
      </main>
      <footer className="w-full border-t py-6">
        <div className="container flex flex-col items-center justify-between gap-4 md:flex-row">
          <div className="flex gap-2 items-center text-lg font-semibold">
            <Shield className="h-5 w-5 text-primary" />
            <span>MalwareGuard AI</span>
          </div>
          <p className="text-center text-sm text-muted-foreground">© 2024 MalwareGuard AI. All rights reserved.</p>
          <div className="flex gap-4">
            <Link href="#" className="text-sm text-muted-foreground underline underline-offset-4">
              Terms
            </Link>
            <Link href="#" className="text-sm text-muted-foreground underline underline-offset-4">
              Privacy
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
