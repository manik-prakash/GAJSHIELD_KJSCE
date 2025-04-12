import {Link} from "react-router-dom";
import {
  Shield,
  Upload,
  FileText,
  AlertTriangle,
  CheckCircle,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Home() {
  return (
      <div className="flex min-h-screen flex-col">
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0">
            <div className="flex gap-2 items-center text-xl font-bold">
              <Shield className="h-6 w-6 text-primary" />
              <span>MalwareGuard AI</span>
            </div>
            <div className="flex flex-1 items-center justify-end space-x-4">
              <nav className="flex items-center space-x-2">
                <Button>
                  <Link to="/dashboard">Dashboard</Link>
                </Button>
                <Button>
                  <Link to="/history">History</Link>
                </Button>
                <Button>
                  <Link to="/settings">Settings</Link>
                </Button>
                <Button>
                  <Link to="/scan">New Scan</Link>
                </Button>
              </nav>
            </div>
          </div>
        </header>
        <main className="flex-1">
          <section className="w-full py-12 md:py-24 lg:py-32 custom-blue-gradient text-white">
            <div className="container px-4 md:px-6">
              <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
                <div className="flex flex-col justify-center space-y-4">
                  <div className="space-y-2">
                    <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                      AI-Powered Malware Detection
                    </h1>
                    <p className="max-w-[600px] text-white/80 md:text-xl">
                      Protect your systems with our advanced AI that detects
                      known and unknown threats in real-time.
                    </p>
                  </div>
                  <div className="flex flex-col gap-2 min-[400px]:flex-row">
                    <Button
                      size="lg"
                      className="gap-2 bg-accent text-black hover:bg-accent/90"
                    >
                      <Upload className="h-5 w-5" />
                      Start Scanning
                    </Button>
                    <Button
                      size="lg"
                      variant="outline"
                      className="border-white text-white hover:bg-white/10"
                    >
                      Learn More
                    </Button>
                  </div>
                </div>
                <div className="mx-auto flex w-full items-center justify-center">
                  <Card className="w-full border-0 shadow-lg">
                    <CardHeader>
                      <CardTitle>Quick Scan</CardTitle>
                      <CardDescription>
                        Upload a file to check for malware
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-10 space-y-2 border-secondary/30">
                        <Upload className="h-10 w-10 text-secondary" />
                        <p className="text-sm text-muted-foreground">
                          Drag and drop files here or click to browse
                        </p>
                        <Button variant="secondary" size="sm">
                          Select Files
                        </Button>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <p className="text-xs text-muted-foreground">
                        Supports .exe, .csv, .docx, .bat and other file types
                      </p>
                    </CardFooter>
                  </Card>
                </div>
              </div>
            </div>
          </section>
          <section className="w-full py-12 md:py-24 lg:py-32">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">
                  Advanced Protection Features
                </h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Our AI-powered malware detection system provides comprehensive
                  protection against various threats.
                </p>
              </div>
              <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 lg:gap-8 xl:gap-10 mt-10">
                <Card className="border-accent/20 hover:border-accent/50 transition-colors">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">
                      Real-time Detection
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Instantly analyze files for malicious content using
                      advanced AI algorithms.
                    </p>
                  </CardContent>
                </Card>
                <Card className="border-accent/20 hover:border-accent/50 transition-colors">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">
                      Zero-day Protection
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Identify unknown threats before they can cause damage to
                      your systems.
                    </p>
                  </CardContent>
                </Card>
                <Card className="border-accent/20 hover:border-accent/50 transition-colors">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Batch Processing</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Scan multiple files simultaneously for efficient threat
                      detection.
                    </p>
                  </CardContent>
                </Card>
                <Card className="border-accent/20 hover:border-accent/50 transition-colors">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Detailed Reports</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Get comprehensive analysis reports with actionable
                      insights.
                    </p>
                  </CardContent>
                </Card>
              </div>
            </div>
          </section>
          <section className="w-full py-12 md:py-24 lg:py-32 custom-blue-overlay">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">
                  How It Works
                </h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Our AI-powered system uses advanced machine learning to detect
                  malware with high accuracy.
                </p>
              </div>
              <div className="mx-auto grid max-w-5xl items-center gap-6 py-12 lg:grid-cols-2 lg:gap-12">
                <div className="flex flex-col justify-center space-y-4">
                  <ul className="grid gap-6">
                    <li className="flex items-start gap-4">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                        <Upload className="h-5 w-5" />
                      </div>
                      <div className="grid gap-1">
                        <h3 className="text-xl font-bold">Upload Files</h3>
                        <p className="text-muted-foreground">
                          Upload individual files or batch process multiple
                          files at once.
                        </p>
                      </div>
                    </li>
                    <li className="flex items-start gap-4">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-secondary/10 text-secondary">
                        <FileText className="h-5 w-5" />
                      </div>
                      <div className="grid gap-1">
                        <h3 className="text-xl font-bold">AI Analysis</h3>
                        <p className="text-muted-foreground">
                          Our AI engine analyzes file structure, behavior
                          patterns, and code signatures.
                        </p>
                      </div>
                    </li>
                    <li className="flex items-start gap-4">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-primary">
                        <CheckCircle className="h-5 w-5" />
                      </div>
                      <div className="grid gap-1">
                        <h3 className="text-xl font-bold">Threat Detection</h3>
                        <p className="text-muted-foreground">
                          Receive instant results with detailed information
                          about detected threats.
                        </p>
                      </div>
                    </li>
                  </ul>
                </div>
                <div className="mx-auto flex w-full items-center justify-center">
                  <Tabs defaultValue="upload" className="w-full max-w-md">
                    <TabsList className="grid w-full grid-cols-3 bg-primary/10">
                      <TabsTrigger
                        value="upload"
                        className="data-[state=active]:bg-primary data-[state=active]:text-white"
                      >
                        Upload
                      </TabsTrigger>
                      <TabsTrigger
                        value="analyze"
                        className="data-[state=active]:bg-primary data-[state=active]:text-white"
                      >
                        Analyze
                      </TabsTrigger>
                      <TabsTrigger
                        value="results"
                        className="data-[state=active]:bg-primary data-[state=active]:text-white"
                      >
                        Results
                      </TabsTrigger>
                    </TabsList>
                    <TabsContent
                      value="upload"
                      className="border rounded-lg p-4 mt-2 border-primary/20"
                    >
                      <div className="flex flex-col items-center justify-center space-y-4 p-4">
                        <Upload className="h-16 w-16 text-secondary" />
                        <p className="text-center text-sm text-muted-foreground">
                          Drag and drop files here or click to browse
                        </p>
                      </div>
                    </TabsContent>
                    <TabsContent
                      value="analyze"
                      className="border rounded-lg p-4 mt-2 border-primary/20"
                    >
                      <div className="flex flex-col items-center justify-center space-y-4 p-4">
                        <div className="h-16 w-16 rounded-full border-4 border-t-secondary border-secondary/20 animate-spin" />
                        <p className="text-center text-sm text-muted-foreground">
                          AI analyzing file structure and behavior patterns...
                        </p>
                      </div>
                    </TabsContent>
                    <TabsContent
                      value="results"
                      className="border rounded-lg p-4 mt-2 border-primary/20"
                    >
                      <div className="flex flex-col items-center justify-center space-y-4 p-4">
                        <div className="flex items-center justify-center">
                          <AlertTriangle className="h-16 w-16 text-red-500" />
                        </div>
                        <h3 className="text-lg font-bold text-red-500">
                          Malware Detected
                        </h3>
                        <p className="text-center text-sm text-muted-foreground">
                          Trojan detected in file. We recommend immediate
                          deletion.
                        </p>
                      </div>
                    </TabsContent>
                  </Tabs>
                </div>
              </div>
            </div>
          </section>
        </main>
        <footer className="w-full border-t py-6 bg-primary text-white">
          <div className="container flex flex-col items-center justify-between gap-4 md:flex-row">
            <div className="flex gap-2 items-center text-lg font-semibold">
              <Shield className="h-5 w-5 text-accent" />
              <span>MalwareGuard AI</span>
            </div>
            <p className="text-center text-sm text-white/70">
              © 2024 MalwareGuard AI. All rights reserved.
            </p>
            <div className="flex gap-4">
              <Link
                to="#"
                className="text-sm text-white/70 underline underline-offset-4 hover:text-white"
              >
                Terms
              </Link>
              <Link
                to="#"
                className="text-sm text-white/70 underline underline-offset-4 hover:text-white"
              >
                Privacy
              </Link>
            </div>
          </div>
        </footer>
      </div>
  );
}
