#include <bits/stdc++.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <edgelist.txt>\n", argv[0]);
        return 1;
    }
    string inPath = argv[1];
    int fd = open(inPath.c_str(), O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat sb;
    fstat(fd, &sb);
    size_t fsize = sb.st_size;
    char* data = (char*)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(data, fsize, MADV_SEQUENTIAL);

    vector<uint32_t> src, dst;
    size_t estEdges = fsize / 10 + 1024;
    src.reserve(estEdges);
    dst.reserve(estEdges);

    uint32_t maxNode = 0;
    size_t i = 0;
    while (i < fsize) {
        if (data[i] == '#' || data[i] == '%') {
            while (i < fsize && data[i] != '\n') i++;
            i++;
            continue;
        }
        if (data[i] == '\n' || data[i] == '\r') { i++; continue; }

        uint32_t a = 0;
        while (i < fsize && data[i] >= '0' && data[i] <= '9') {
            a = a * 10 + (data[i] - '0');
            i++;
        }
        while (i < fsize && (data[i] == ' ' || data[i] == '\t')) i++;

        uint32_t b = 0;
        while (i < fsize && data[i] >= '0' && data[i] <= '9') {
            b = b * 10 + (data[i] - '0');
            i++;
        }
        while (i < fsize && data[i] != '\n') i++;
        i++;

        src.push_back(a);
        dst.push_back(b);
        if (a > maxNode) maxNode = a;
        if (b > maxNode) maxNode = b;
    }

    munmap(data, fsize);
    close(fd);

    size_t E = src.size();
    uint32_t V = maxNode + 1;
    fprintf(stderr, "Parsed E=%zu edges, V=%u vertices\n", E, V);

    // counting sort into CSR
    vector<uint64_t> rowPtr(V + 1, 0);
    for (size_t e = 0; e < E; e++) rowPtr[src[e] + 1]++;
    for (uint32_t v = 0; v < V; v++) rowPtr[v + 1] += rowPtr[v];

    vector<uint32_t> col(E);
    vector<uint64_t> fillPos(rowPtr.begin(), rowPtr.end() - 1);
    for (size_t e = 0; e < E; e++) {
        uint32_t s = src[e];
        col[fillPos[s]++] = dst[e];
    }

    vector<uint32_t>().swap(src);
    vector<uint32_t>().swap(dst);

    string outPath = inPath;
    size_t dotPos = outPath.find_last_of('.');
    string base = (dotPos == string::npos) ? outPath : outPath.substr(0, dotPos);
    string ext = (dotPos == string::npos) ? "" : outPath.substr(dotPos);
    string outFile = base + "_csr" + (ext.empty() ? ".txt" : ext);

    FILE* fout = fopen(outFile.c_str(), "w");
    if (!fout) { perror("fopen"); return 1; }

    const size_t BUF_CAP = 1 << 24; // 16MB
    vector<char> buf(BUF_CAP);
    size_t bp = 0;

    auto flushBuf = [&]() { fwrite(buf.data(), 1, bp, fout); bp = 0; };

    auto writeUint = [&](uint64_t val) {
        char tmp[24];
        int len = 0;
        if (val == 0) tmp[len++] = '0';
        else while (val > 0) { tmp[len++] = '0' + (val % 10); val /= 10; }
        if (bp + len + 1 > BUF_CAP) flushBuf();
        while (len > 0) buf[bp++] = tmp[--len];
    };
    auto writeChar = [&](char c) {
        if (bp + 1 > BUF_CAP) flushBuf();
        buf[bp++] = c;
    };

    // line 1: E V source
    writeUint(E); writeChar(' ');
    writeUint(V); writeChar(' ');
    writeUint(0); writeChar('\n');

    // line 2: col
    for (size_t e = 0; e < E; e++) {
        writeUint(col[e]);
        writeChar(e + 1 < E ? ' ' : '\n');
    }

    // line 3: row_ptr (V entries, final E omitted)
    for (uint32_t v = 0; v < V; v++) {
        writeUint(rowPtr[v]);
        writeChar(v + 1 < V ? ' ' : '\n');
        // if(v == V-1)
        // cout<<rowPtr[v]<<endl; 
    }

    flushBuf();
    fclose(fout);
    fprintf(stderr, "Done. Wrote %s\n", outFile.c_str());
    return 0;
}