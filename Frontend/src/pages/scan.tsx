import { useState } from "react"
import { Shield, Upload, FileText, AlertTriangle, CheckCircle, X, Info, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {Link} from "react-router-dom"

export default function ScanPage() {
  const [files, setFiles] = useState<File[]>([])
  const [scanning, setScanning] = useState(false)
  const [scanComplete, setScanComplete] = useState(false)
  const [progress, setProgress] = useState(0)

  const scanResults = [
    {
      name: "document.docx",
      size: "245 KB",
      status: "safe",
      type: "Document",
      confidence: 98,
    },
    {
      name: "setup.exe",
      size: "1.2 MB",
      status: "malware",
      type: "Executable",
      threat: "Trojan.Win32.Miner",
      confidence: 96,
    },
    {
      name: "data.csv",
      size: "56 KB",
      status: "safe",
      type: "Data File",
      confidence: 99,
    },
    {
      name: "script.bat",
      size: "12 KB",
      status: "suspicious",
      type: "Script",
      threat: "Potentially unwanted behavior",
      confidence: 78,
    },
  ]

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      setFiles(Array.from(files));
    }
  }

  const handleDrop = (e: { preventDefault: () => void; dataTransfer: { files: Iterable<File> | ArrayLike<File> } }) => {
    e.preventDefault()
    if (e.dataTransfer.files) {
      setFiles(Array.from(e.dataTransfer.files))
    }
  }

  const handleDragOver = (e: { preventDefault: () => void }) => {
    e.preventDefault()
  }

  const startScan = () => {
    if (files.length === 0) return

    setScanning(true)
    setScanComplete(false)
    setProgress(0)

    // Simulate scanning progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setScanning(false)
          setScanComplete(true)
          return 100
        }
        return prev + 10
      })
    }, 500)
  }

  const resetScan = () => {
    setFiles([])
    setScanning(false)
    setScanComplete(false)
    setProgress(0)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "safe":
        return "bg-primary"
      case "malware":
        return "bg-red-500"
      case "suspicious":
        return "bg-yellow-500"
      default:
        return "bg-slate-500"
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "safe":
        return <Badge className="bg-primary hover:bg-primary/80">Safe</Badge>
      case "malware":
        return <Badge className="bg-red-500 hover:bg-red-600">Malware</Badge>
      case "suspicious":
        return <Badge className="bg-yellow-500 hover:bg-yellow-600">Suspicious</Badge>
      default:
        return <Badge>Unknown</Badge>
    }
  }

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0">
          <Link to="/" className="flex gap-2 items-center text-xl font-bold">
            <Shield className="h-6 w-6 text-primary" />
            <span>MalwareGuard AI</span>
          </Link>
          <div className="flex flex-1 items-center justify-end space-x-4">
            <nav className="flex items-center space-x-2">
              <Button asChild variant="ghost">
                <Link to="/dashboard">Dashboard</Link>
              </Button>
              <Button asChild variant="ghost">
                <Link to="/history">History</Link>
              </Button>
              <Button asChild variant="ghost">
                <Link to="/settings">Settings</Link>
              </Button>
              <Button asChild variant="default">
                <Link to="/scan">New Scan</Link>
              </Button>
            </nav>
          </div>
        </div>
      </header>
      <main className="flex-1 container py-8">
        <div className="flex flex-col gap-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">Malware Scan</h1>
            <p className="text-muted-foreground">Upload files to scan for malware and other threats</p>
          </div>

          <Tabs defaultValue="single" className="w-full">
            <TabsList>
              <TabsTrigger value="single">Single File</TabsTrigger>
              <TabsTrigger value="batch">Batch Scan</TabsTrigger>
              <TabsTrigger value="folder">Folder Scan</TabsTrigger>
            </TabsList>
            <TabsContent value="single" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Single File Scan</CardTitle>
                  <CardDescription>Upload a single file to check for malware</CardDescription>
                </CardHeader>
                <CardContent>
                  {!scanComplete ? (
                    <div
                      className={`flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-10 space-y-4 ${
                        files.length > 0 ? "border-primary bg-primary/10" : ""
                      }`}
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                    >
                      {files.length === 0 ? (
                        <>
                          <Upload className="h-16 w-16 text-muted-foreground" />
                          <p className="text-center text-muted-foreground">
                            Drag and drop a file here or click to browse
                          </p>
                          <input type="file" id="file-upload" className="hidden" onChange={handleFileChange} />
                          <Button variant="secondary" onClick={() => document.getElementById("file-upload")?.click()}>
                            Select File
                          </Button>
                        </>
                      ) : (
                        <>
                          <FileText className="h-16 w-16 text-primary" />
                          <div className="text-center">
                            <p className="font-medium">{files[0].name}</p>
                            <p className="text-sm text-muted-foreground">{(files[0].size / 1024).toFixed(2)} KB</p>
                          </div>
                          <div className="flex gap-2">
                            <Button variant="outline" size="sm" onClick={resetScan}>
                              <X className="h-4 w-4 mr-1" /> Remove
                            </Button>
                            <Button size="sm" onClick={startScan} disabled={scanning}>
                              {scanning ? (
                                <>
                                  <Loader2 className="h-4 w-4 mr-1 animate-spin" /> Scanning...
                                </>
                              ) : (
                                <>
                                  <Shield className="h-4 w-4 mr-1" /> Scan Now
                                </>
                              )}
                            </Button>
                          </div>
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="flex items-start gap-4">
                        <div className={`p-2 rounded-full ${getStatusColor(scanResults[1].status)}`}>
                          {scanResults[1].status === "safe" ? (
                            <CheckCircle className="h-6 w-6 text-white" />
                          ) : scanResults[1].status === "malware" ? (
                            <AlertTriangle className="h-6 w-6 text-white" />
                          ) : (
                            <Info className="h-6 w-6 text-white" />
                          )}
                        </div>
                        <div className="flex-1">
                          <div className="flex justify-between items-start">
                            <div>
                              <h3 className="font-semibold text-lg">{scanResults[1].name}</h3>
                              <p className="text-sm text-muted-foreground">
                                {scanResults[1].size} • {scanResults[1].type}
                              </p>
                            </div>
                            {getStatusBadge(scanResults[1].status)}
                          </div>

                          {scanResults[1].status !== "safe" && (
                            <Alert
                              className="mt-4"
                              variant={scanResults[1].status === "malware" ? "destructive" : "default"}
                            >
                              <AlertTriangle className="h-4 w-4" />
                              <AlertTitle>Threat Detected</AlertTitle>
                              <AlertDescription>
                                {scanResults[1].threat} (Confidence: {scanResults[1].confidence}%)
                              </AlertDescription>
                            </Alert>
                          )}

                          <div className="mt-4 grid grid-cols-2 gap-2">
                            <Button variant="outline" onClick={resetScan}>
                              Scan Another File
                            </Button>
                            <Button variant="default">View Detailed Report</Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {scanning && (
                    <div className="mt-6 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Scanning file...</span>
                        <span>{progress}%</span>
                      </div>
                      <Progress value={progress} className="h-2" />
                      <p className="text-xs text-muted-foreground mt-2">
                        AI is analyzing file structure, behavior patterns, and code signatures
                      </p>
                    </div>
                  )}
                </CardContent>
                <CardFooter className="flex justify-between border-t pt-4">
                  <p className="text-xs text-muted-foreground">Supports .exe, .csv, .docx, .bat and other file types</p>
                  {!scanning && !scanComplete && files.length > 0 && <Button onClick={startScan}>Start Scan</Button>}
                </CardFooter>
              </Card>
            </TabsContent>
            <TabsContent value="batch" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Batch File Scan</CardTitle>
                  <CardDescription>Upload multiple files to check for malware</CardDescription>
                </CardHeader>
                <CardContent>
                  {!scanComplete ? (
                    <div
                      className="flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-10 space-y-4"
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                    >
                      <Upload className="h-16 w-16 text-muted-foreground" />
                      <p className="text-center text-muted-foreground">
                        Drag and drop multiple files here or click to browse
                      </p>
                      <input type="file" id="batch-upload" className="hidden" multiple onChange={handleFileChange} />
                      <Button variant="secondary" onClick={() => document.getElementById("batch-upload")?.click()}>
                        Select Files
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="flex justify-between items-center">
                        <h3 className="font-semibold">Scan Results</h3>
                        <div className="flex gap-2 text-sm">
                          <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full bg-primary"></div>
                            <span>Safe: 2</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                            <span>Suspicious: 1</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full bg-red-500"></div>
                            <span>Malware: 1</span>
                          </div>
                        </div>
                      </div>

                      <div className="border rounded-lg overflow-hidden">
                        <div className="grid grid-cols-12 gap-4 p-3 bg-muted font-medium text-sm">
                          <div className="col-span-5">Filename</div>
                          <div className="col-span-2">Size</div>
                          <div className="col-span-2">Type</div>
                          <div className="col-span-2">Status</div>
                          <div className="col-span-1">Actions</div>
                        </div>

                        {scanResults.map((result, index) => (
                          <div key={index} className="grid grid-cols-12 gap-4 p-3 border-t items-center text-sm">
                            <div className="col-span-5 font-medium">{result.name}</div>
                            <div className="col-span-2 text-muted-foreground">{result.size}</div>
                            <div className="col-span-2 text-muted-foreground">{result.type}</div>
                            <div className="col-span-2">{getStatusBadge(result.status)}</div>
                            <div className="col-span-1">
                              <Button variant="ghost" size="icon">
                                <Info className="h-4 w-4" />
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="flex justify-end gap-2">
                        <Button variant="outline" onClick={resetScan}>
                          Scan More Files
                        </Button>
                        <Button>Download Report</Button>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="folder" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Folder Scan</CardTitle>
                  <CardDescription>Upload an entire folder to check for malware</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-10 space-y-4">
                    <Upload className="h-16 w-16 text-muted-foreground" />
                    <p className="text-center text-muted-foreground">Select a folder to scan all files inside</p>
                    <Button variant="secondary">Select Folder</Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <Card>
            <CardHeader>
              <CardTitle>Scan Settings</CardTitle>
              <CardDescription>Configure how the AI analyzes your files</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                <div className="flex items-start space-x-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/20 text-primary dark:bg-primary/20">
                    <Shield className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-medium">Scan Depth</h3>
                    <p className="text-sm text-muted-foreground">
                      Deep scan analyzes file contents and behavior patterns
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/20 text-primary dark:bg-primary/20">
                    <FileText className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-medium">File Types</h3>
                    <p className="text-sm text-muted-foreground">
                      Scan all file types including executables and documents
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/20 text-primary dark:bg-primary/20">
                    <AlertTriangle className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-medium">Threat Detection</h3>
                    <p className="text-sm text-muted-foreground">
                      Detect viruses, trojans, ransomware, and other malware
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
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
            <Link to="#" className="text-sm text-muted-foreground underline underline-offset-4">
              Terms
            </Link>
            <Link to="#" className="text-sm text-muted-foreground underline underline-offset-4">
              Privacy
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
