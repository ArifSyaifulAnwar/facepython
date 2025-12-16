#!/usr/bin/env python3
"""
Script untuk mengkonversi FILE_BYTE (base64) dari database menjadi file PDF
Mendukung PDF dengan password
"""

import base64
import sys
import os
from pathlib import Path

def convert_base64_to_pdf(base64_string, output_filename, password=None):
    """
    Konversi base64 string menjadi file PDF
    
    Args:
        base64_string: String base64 dari database
        output_filename: Nama file output (akan ditambahkan .pdf otomatis)
        password: Password PDF jika ada (optional)
    """
    try:
        # Bersihkan whitespace dari base64 string
        clean_base64 = base64_string.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        print(f"📊 Panjang base64: {len(clean_base64)} karakter")
        
        # Decode base64 ke bytes
        pdf_bytes = base64.b64decode(clean_base64)
        
        print(f"📦 Ukuran file: {len(pdf_bytes)} bytes ({len(pdf_bytes)/1024:.2f} KB)")
        
        # Validasi PDF signature (harus dimulai dengan %PDF)
        if not pdf_bytes.startswith(b'%PDF'):
            print("❌ Error: Data bukan format PDF yang valid!")
            print(f"   Header yang ditemukan: {pdf_bytes[:10]}")
            return False
        
        # Tambahkan ekstensi .pdf jika belum ada
        if not output_filename.endswith('.pdf'):
            output_filename += '.pdf'
        
        # Tulis ke file
        output_path = Path(output_filename)
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
        
        print(f"✅ File PDF berhasil dibuat: {output_path.absolute()}")
        
        if password:
            print(f"🔒 PDF ini TERPROTEKSI dengan password!")
            print(f"   Gunakan password: '{password}' untuk membuka file")
        else:
            print("🔓 PDF tanpa password")
        
        # Informasi tambahan
        print(f"\n📍 Lokasi file: {output_path.absolute()}")
        print(f"📏 Ukuran file: {output_path.stat().st_size / 1024:.2f} KB")
        
        return True
        
    except base64.binascii.Error as e:
        print(f"❌ Error decode base64: {e}")
        print("   Pastikan data base64 lengkap dan benar!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def process_from_file(input_file, output_filename, password=None):
    """
    Proses file yang berisi base64 string
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            base64_string = f.read()
        
        print(f"📂 Membaca dari file: {input_file}")
        return convert_base64_to_pdf(base64_string, output_filename, password)
        
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {input_file}")
        return False
    except Exception as e:
        print(f"❌ Error membaca file: {e}")
        return False


def main():
    print("=" * 60)
    print("🔄 PDF Converter - Base64 to PDF")
    print("=" * 60)
    print()
    
    if len(sys.argv) < 2:
        print("📖 Cara penggunaan:")
        print("   1. Dari file:")
        print("      python pdf_converter.py input.txt output.pdf [password]")
        print()
        print("   2. Langsung base64:")
        print("      python pdf_converter.py --direct 'base64_string' output.pdf [password]")
        print()
        print("   3. Mode interaktif:")
        print("      python pdf_converter.py --interactive")
        print()
        return
    
    # Mode interactive
    if sys.argv[1] == '--interactive' or sys.argv[1] == '-i':
        print("🖥️  Mode Interaktif")
        print("-" * 60)
        
        # Input method
        print("\nPilih metode input:")
        print("1. Paste base64 langsung")
        print("2. Dari file")
        choice = input("Pilihan (1/2): ").strip()
        
        if choice == '1':
            print("\n📋 Paste base64 string (tekan Enter 2x untuk selesai):")
            lines = []
            while True:
                line = input()
                if line == '':
                    break
                lines.append(line)
            base64_string = ''.join(lines)
        else:
            input_file = input("\n📂 Path file input: ").strip()
            with open(input_file, 'r', encoding='utf-8') as f:
                base64_string = f.read()
        
        output_filename = input("\n💾 Nama file output (tanpa .pdf): ").strip()
        password = input("🔑 Password PDF (kosongkan jika tidak ada): ").strip()
        
        if not password:
            password = None
        
        convert_base64_to_pdf(base64_string, output_filename, password)
        
    # Mode direct
    elif sys.argv[1] == '--direct' or sys.argv[1] == '-d':
        if len(sys.argv) < 4:
            print("❌ Format salah!")
            print("   Gunakan: python pdf_converter.py --direct 'base64_string' output.pdf [password]")
            return
        
        base64_string = sys.argv[2]
        output_filename = sys.argv[3]
        password = sys.argv[4] if len(sys.argv) > 4 else None
        
        convert_base64_to_pdf(base64_string, output_filename, password)
        
    # Mode file
    else:
        input_file = sys.argv[1]
        output_filename = sys.argv[2] if len(sys.argv) > 2 else 'output.pdf'
        password = sys.argv[3] if len(sys.argv) > 3 else None
        
        process_from_file(input_file, output_filename, password)


if __name__ == '__main__':
    main()