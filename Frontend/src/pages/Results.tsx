import { useLocation } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface EngineResult {
  engine_name: string;
  result: string;
}

interface LocationState {
  engine_results: EngineResult[];
}

function ResultsPage() {
  const location = useLocation();
  const state = location.state as LocationState || { engine_results: [] };
  const { engine_results } = state;

  const [loading, setLoading] = useState(false);
  const [pdfLink, setPdfLink] = useState<string | null>(null);
  const [error, setError] = useState<string>('');

  const handleGeneratePDF = async () => {
    try {
      setLoading(true);
      setError('');
      setPdfLink(null);

      const response = await axios.post(
        'http://localhost:5000/generate-pdf',
        { engine_results },
        { responseType: 'blob' }
      );

      const pdfBlob = new Blob([response.data], { type: 'application/pdf' });
      const pdfUrl = URL.createObjectURL(pdfBlob);

      setPdfLink(pdfUrl);
    } catch (err) {
      console.error(err);
      setError('Failed to generate PDF.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 flex flex-col items-center justify-center">
      <Card className="w-full max-w-4xl shadow-md">
        <CardHeader>
          <CardTitle className="text-center text-2xl font-bold">Scan Results</CardTitle>
        </CardHeader>
        <CardContent>
          {engine_results.length === 0 ? (
            <p className="text-center text-gray-500">No results found.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Engine Name</TableHead>
                  <TableHead>Result</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {engine_results.map((engine, index) => (
                  <TableRow key={index}>
                    <TableCell>{engine.engine_name}</TableCell>
                    <TableCell>{engine.result}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
          <div className="flex justify-center mt-6">
            <Button onClick={handleGeneratePDF} disabled={loading}>
              {loading ? 'Generating PDF...' : 'Generate PDF'}
            </Button>
          </div>

          {pdfLink && (
            <div className="flex justify-center mt-4">
              <a href={pdfLink} download="scan_results.pdf" className="text-green-600 underline">
                Download PDF
              </a>
            </div>
          )}

          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default ResultsPage;
