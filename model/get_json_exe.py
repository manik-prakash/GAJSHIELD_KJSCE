import lief
import hashlib
import numpy as np
import json
import os
import re
import math
from typing import Dict, List, Any, Optional, Union

class FeatureType:
    """Base class from which each feature type may inherit"""
    name = ''
    dim = 0

    def __repr__(self):
        return f'{self.name}({self.dim})'

    def raw_features(self, bytez, lief_binary):
        """Generate a JSON-able representation of the file"""
        raise NotImplementedError

    def process_raw_features(self, raw_obj):
        """Generate a feature vector from the raw features"""
        raise NotImplementedError

    def feature_vector(self, bytez, lief_binary):
        """Directly calculate the feature vector from the sample itself"""
        return self.process_raw_features(self.raw_features(bytez, lief_binary))


class ByteHistogram(FeatureType):
    """Byte histogram (count + non-normalized) over the entire binary file"""
    name = 'histogram'
    dim = 256

    def __init__(self):
        super().__init__()

    def raw_features(self, bytez, lief_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum_counts = counts.sum()
        normalized = counts / sum_counts if sum_counts > 0 else counts
        return normalized


class ByteEntropyHistogram(FeatureType):
    """2d byte/entropy histogram based loosely on (Saxe and Berlin, 2015)"""
    name = 'byteentropy'
    dim = 256

    def __init__(self, step=1024, window=2048):
        super().__init__()
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        # coarse histogram, 16 bytes per bin
        c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        # * x2 b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4 bits)
        H = np.sum(-p[wh] * np.log2(p[wh])) * 2 if len(wh) > 0 else 0

        Hbin = int(H * 2)  # up to 16 bins (max entropy is 8 bits)
        if Hbin == 16:  # handle entropy = 8.0 bits
            Hbin = 15

        return Hbin, c

    def raw_features(self, bytez, lief_binary):
        output = np.zeros((16, 16), dtype=np.int8)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            # strided trick for rolling windows
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]

            # from the blocks, compute histogram
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c

        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum_counts = counts.sum()
        normalized = counts / sum_counts if sum_counts > 0 else counts
        return normalized


class StringExtractor(FeatureType):
    """Extract strings from the binary and compute some statistics about them"""
    name = 'strings'
    dim = 1

    def __init__(self, min_length=4):
        super().__init__()
        self.min_length = min_length

    def raw_features(self, bytez, lief_binary):
        string_list = []
        
        # Extract ASCII strings
        ascii_regex = b'[ -~]{%d,}' % self.min_length
        string_list.extend(re.findall(ascii_regex, bytez))
        
        # Extract UTF-16 strings
        utf16_regex = b'(?:[ -~]\x00){%d,}' % self.min_length
        utf16_strings = re.findall(utf16_regex, bytez)
        string_list.extend([s.decode('utf-16le', errors='ignore').encode('utf-8') for s in utf16_strings])
        
        # Decode strings and compute stats
        decoded_strings = [s.decode('utf-8', errors='ignore') for s in string_list]
        
        # Calculate statistics
        paths = 0
        urls = 0
        registry = 0
        mz = 0
        
        # Simple patterns for counting
        for string in decoded_strings:
            if any(p in string.lower() for p in [':\\', '/', '.dll', '.exe', '.sys']):
                paths += 1
            if any(p in string.lower() for p in ['http://', 'https://', 'ftp://', 'www.']):
                urls += 1
            if 'HKEY_' in string or 'CurrentVersion' in string:
                registry += 1
            if 'MZ' in string:
                mz += 1

        # Calculate entropy
        entropy = 0
        if decoded_strings:
            all_strings = ''.join(decoded_strings)
            if all_strings:
                character_counts = {}
                for char in all_strings:
                    if char in character_counts:
                        character_counts[char] += 1
                    else:
                        character_counts[char] = 1
                
                total_chars = len(all_strings)
                entropy = 0
                for count in character_counts.values():
                    p = count / total_chars
                    entropy -= p * math.log2(p)

        return {
            'entropy': entropy,
            'paths': paths,
            'urls': urls,
            'registry': registry,
            'MZ': mz
        }

    def process_raw_features(self, raw_obj):
        return np.array([raw_obj['entropy']], dtype=np.float32)


class GeneralFileInfo(FeatureType):
    """General information about the file"""
    name = 'general'
    dim = 1

    def __init__(self):
        super().__init__()

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {
                'size': len(bytez),
                'vsize': 0,
                'exports': 0,
                'imports': 0,
            }
        
        return {
            'size': len(bytez),
            'vsize': sum(section.virtual_size for section in lief_binary.sections),
            'exports': len(lief_binary.exported_functions) if lief_binary.has_exports else 0,
            'imports': sum(len(library.entries) for library in lief_binary.imports),
        }

    def process_raw_features(self, raw_obj):
        return np.array([
            raw_obj['size'], raw_obj['vsize'], raw_obj['exports'], raw_obj['imports']
        ], dtype=np.float32)


class OptionalHeaderInfo(FeatureType):
    """Optional header information from the PE file"""
    name = 'optional'
    dim = 1

    def __init__(self):
        super().__init__()

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {
                'subsystem': 'UNKNOWN',
                'dll_characteristics': [],
                'magic': 'UNKNOWN',
            }
        
        optional_header = lief_binary.optional_header
        
        dll_characteristics = []
        if hasattr(optional_header, 'has_dll_characteristics'):
            for char_name, char_flag in lief.PE.DLL_CHARACTERISTICS.__members__.items():
                if optional_header.has_dll_characteristic(char_flag):
                    dll_characteristics.append(char_name)
        
        magic = 'UNKNOWN'
        if hasattr(lief.PE, 'PE_TYPE'):
            if optional_header.magic == lief.PE.PE_TYPE.PE32:
                magic = 'PE32'
            elif optional_header.magic == lief.PE.PE_TYPE.PE32_PLUS:
                magic = 'PE32+'
        
        subsystem_name = str(optional_header.subsystem).split('.')[-1]
        
        return {
            'subsystem': subsystem_name,
            'dll_characteristics': dll_characteristics,
            'magic': magic,
        }

    def process_raw_features(self, raw_obj):
        # Just a placeholder since this is mainly for JSON output
        return np.array([0], dtype=np.float32)


class SectionInfo(FeatureType):
    """Information about the sections in the PE file"""
    name = 'section'
    dim = 1

    def __init__(self):
        super().__init__()

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {
                'entry': '',
                'sections': []
            }
        
        # Find the section containing the entry point
        entry_section = ''
        entry_addr = lief_binary.optional_header.addressof_entrypoint
        
        for section in lief_binary.sections:
            if section.virtual_address <= entry_addr < (section.virtual_address + section.virtual_size):
                entry_section = section.name
                break
        
        sections_info = []
        for section in lief_binary.sections:
            # Calculate entropy
            entropy = 0
            if hasattr(section, 'content') and section.content:
                content = np.array(section.content)
                counts = np.bincount(content, minlength=256)
                p = counts / float(len(content))
                valid_p = p[p > 0]
                entropy = np.sum(-valid_p * np.log2(valid_p))
            
            # Get section properties
            props = []
            if hasattr(lief.PE, 'SECTION_CHARACTERISTICS'):
                if section.has_characteristic(lief.PE.SECTION_CHARACTERISTICS.CNT_CODE):
                    props.append('CNT_CODE')
                if section.has_characteristic(lief.PE.SECTION_CHARACTERISTICS.CNT_INITIALIZED_DATA):
                    props.append('CNT_INITIALIZED_DATA')
                if section.has_characteristic(lief.PE.SECTION_CHARACTERISTICS.CNT_UNINITIALIZED_DATA):
                    props.append('CNT_UNINITIALIZED_DATA')
                if section.has_characteristic(lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE):
                    props.append('MEM_EXECUTE')
                if section.has_characteristic(lief.PE.SECTION_CHARACTERISTICS.MEM_READ):
                    props.append('MEM_READ')
                if section.has_characteristic(lief.PE.SECTION_CHARACTERISTICS.MEM_WRITE):
                    props.append('MEM_WRITE')
            
            sections_info.append({
                'name': section.name,
                'size': section.size,
                'entropy': entropy,
                'vsize': section.virtual_size,
                'props': props
            })
        
        return {
            'entry': entry_section,
            'sections': sections_info
        }

    def process_raw_features(self, raw_obj):
        # Just a placeholder since this is mainly for JSON output
        return np.array([0], dtype=np.float32)


class PEFFileExtractor:
    """Extract features from a PE file and output them in a JSON format"""
    
    def __init__(self):
        self.features = [
            ByteHistogram(),
            ByteEntropyHistogram(),
            StringExtractor(),
            GeneralFileInfo(),
            OptionalHeaderInfo(),
            SectionInfo()
        ]
    
    def extract_features(self, file_path):
        """Extract features from a PE file"""
        with open(file_path, 'rb') as f:
            bytez = f.read()
        
        # Calculate basic hash values
        sha256 = hashlib.sha256(bytez).hexdigest()
        md5 = hashlib.md5(bytez).hexdigest()
        
        # Parse the PE file with lief
        try:
            lief_binary = lief.parse(bytez)
            if not isinstance(lief_binary, lief.PE.Binary):
                print("Not a valid PE file")
                lief_binary = None
        except Exception as e:
            print(f"Error parsing PE file: {str(e)}")
            lief_binary = None
        
        # Extract all features
        features = {
            "sha256": sha256,
            "md5": md5,
            "label": 0,  # Default label, can be set later
            "avclass": ""  # Default avclass, can be set later
        }
        
        # Extract features from each feature extractor
        for fe in self.features:
            features[fe.name] = fe.raw_features(bytez, lief_binary)
        
        return features
    
    def save_features(self, features, output_file):
        """Save extracted features to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pef_extractor.py <pe_file_path> [output_file]")
        sys.exit(1)
    
    pe_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"{os.path.basename(pe_file)}.json"
    
    extractor = PEFFileExtractor()
    try:
        features = extractor.extract_features(pe_file)
        extractor.save_features(features, output_file)
        print(f"Features extracted and saved to {output_file}")
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        sys.exit(1)