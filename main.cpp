#include <boost/core/demangle.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iostream>
#include <libconfig.h++>
#include <map>
#include <regex>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <string>
#include <typeinfo>
#include <unistd.h>
#include <vector>

template <class T>
struct X {
};

int test()
{
    std::string name = "_ZSt4fabsf";
    std::cout << name << std::endl;                                // prints 1XIiE
    std::cout << boost::core::demangle(name.c_str()) << std::endl; // prints X<int>
    name = "_ZNSt14error_categoryD2Ev@plt";
    std::cout << name << std::endl;                                // prints 1XIiE
    std::cout << boost::core::demangle(name.c_str()) << std::endl; // prints X<int>
    return 0;
}

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.
using namespace libconfig;
using namespace std;
int config(int argc, char** argv)
{
    Config cfg;

    // Read the file. If there is an error, report it and exit.
    try {
        cfg.readFile(argv[1]);
    } catch (const FileIOException& fioex) {
        std::cerr << "I/O error while reading file." << std::endl;
        return (EXIT_FAILURE);
    } catch (const ParseException& pex) {
        std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                  << " - " << pex.getError() << std::endl;
        return (EXIT_FAILURE);
    }

    // Get the store name.
    try {
        string name = cfg.lookup("name");
        cout << "Store name: " << name << endl
             << endl;
    } catch (const SettingNotFoundException& nfex) {
        cerr << "No 'name' setting in configuration file." << endl;
    }

    const Setting& root = cfg.getRoot();

    // Output a list of all books in the inventory.
    try {
        const Setting& books = root["inventory"]["books"];
        int count = books.getLength();

        cout << setw(30) << left << "TITLE"
             << "  "
             << setw(30) << left << "AUTHOR"
             << "   "
             << setw(6) << left << "PRICE"
             << "  "
             << "QTY"
             << endl;

        for (int i = 0; i < count; ++i) {
            const Setting& book = books[i];

            // Only output the record if all of the expected fields are present.
            string title, author;
            double price;
            int qty;

            if (!(book.lookupValue("title", title)
                    && book.lookupValue("author", author)
                    && book.lookupValue("price", price)
                    && book.lookupValue("qty", qty)))
                continue;

            cout << setw(30) << left << title << "  "
                 << setw(30) << left << author << "  "
                 << '$' << setw(6) << right << price << "  "
                 << qty
                 << endl;
        }
        cout << endl;
    } catch (const SettingNotFoundException& nfex) {
        // Ignore.
    }

    // Output a list of all books in the inventory.
    try {
        const Setting& movies = root["inventory"]["movies"];
        int count = movies.getLength();

        cout << setw(30) << left << "TITLE"
             << "  "
             << setw(10) << left << "MEDIA"
             << "   "
             << setw(6) << left << "PRICE"
             << "  "
             << "QTY"
             << endl;

        for (int i = 0; i < count; ++i) {
            const Setting& movie = movies[i];

            // Only output the record if all of the expected fields are present.
            string title, media;
            double price;
            int qty;

            if (!(movie.lookupValue("title", title)
                    && movie.lookupValue("media", media)
                    && movie.lookupValue("price", price)
                    && movie.lookupValue("qty", qty)))
                continue;

            cout << setw(30) << left << title << "  "
                 << setw(10) << left << media << "  "
                 << '$' << setw(6) << right << price << "  "
                 << qty
                 << endl;
        }
        cout << endl;
    } catch (const SettingNotFoundException& nfex) {
        // Ignore.
    }

    return 0;
}

class TraceRecord {
public:
    int pid;
    double timestamp;
    uint64_t cycles;
    uint64_t addr;
    char proc_name[1000];
    char func_name[1000];
    void dump();

private:
};

void TraceRecord::dump()
{
    printf("proc_name:%s, pid:%d, timestamp:%lf, cycles:%llu addr:%llu, function:%s\n", proc_name, pid, timestamp, cycles, addr, func_name);
}

void dump_by(auto* pFileReport, auto& vec_ltr, auto& kf_map, auto filter, auto rgb, auto offset, auto bTraceAll)
{
    uint64_t count = 0, downsample = 1;

    fprintf(pFileReport, "\n{");

    fprintf(pFileReport, "name: '%s',", filter);

    fprintf(pFileReport, "color: '%s',", rgb);

    fprintf(pFileReport, "turboThreshold: %u, ", vec_ltr.size());

    fprintf(pFileReport, "data: [\n");

    for (std::vector<TraceRecord>::iterator it = vec_ltr.begin(); it != vec_ltr.end(); it++) {
        if ((count++) % downsample == 0) {
            int id_offset = 0;
            std::string key((*it).func_name);
            if (key.find(filter) != std::string::npos || bTraceAll) {
                id_offset = offset;
                fprintf(pFileReport, "{ x: %lf, y: %d, name: \"%s\"},\n", (*it).timestamp, kf_map[key] + id_offset, key.c_str());
            }
        }
    }
    fprintf(pFileReport, "]},\n");
}

int main(int argc, char* argv[])
{
    FILE *pFile, *pFileReport;
    char mystring[6000];
    std::vector<TraceRecord> vec_ltr;

    int t = 0;
    if (argc < 3) {
        printf("Usage: ./fsa filter.cfg perf.script\n");
        return -1;
    }

    config(argc, argv);

    pFile = fopen(argv[2], "r");
    if (pFile == NULL) {
        perror("Error opening file");
    } else {
        while (fgets(mystring, sizeof(mystring), pFile) != NULL) {
            // e.g.  [00:57:32.541424556] (+0.000000901) paslab-cyliu
            // e.g. std::regex rgx(".*FILE_(\\w+)_EVENT\\.DAT.*");
            TraceRecord tr;
            char str_tmp[2000];
            uint64_t timestamp;
            sscanf(mystring, "%s %d %s %lf: %d %s %x %s %s\n", tr.proc_name,
                &tr.pid,
                str_tmp,
                &tr.timestamp,
                &tr.cycles,
                str_tmp,
                &tr.addr,
                tr.func_name,
                str_tmp);
            //ltr_tmp.kf_name = boost::core::demangle( func_name );
            //tr.dump();
            vec_ltr.push_back(tr);
        }
        fclose(pFile);
    }

    std::map<std::string, int> kf_map;
    // show content:
    for (std::vector<TraceRecord>::iterator it = vec_ltr.begin(); it != vec_ltr.end(); ++it) {
        std::string key((*it).func_name);
        kf_map[key] = 1;
    }

    int kf_id = 0;
    for (std::map<std::string, int>::iterator it = kf_map.begin(); it != kf_map.end(); ++it) {
        (*it).second = kf_id += 10;
    }

    pFileReport = fopen("trace.csv", "w");
    fprintf(pFileReport, "timestamp,func_id,func_name\n");
    for (std::vector<TraceRecord>::iterator it = vec_ltr.begin(); it != vec_ltr.end(); it++) {
        std::string key((*it).func_name);
        fprintf(pFileReport, "%lf,%d,%s\n", (*it).timestamp, kf_map[key], key.c_str());
    }
    fclose(pFileReport);

    pFileReport = fopen("data.js", "w");
    fprintf(pFileReport, "trace_data = [");
    dump_by(pFileReport, vec_ltr, kf_map, "none", "rgba(223,83,83,.5)", 0, true);
    fprintf(pFileReport, "\n];");
    fclose(pFileReport);

    return 0;
}
