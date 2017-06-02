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

class Filter {
public:
    std::string tracing_mode;
    std::vector<std::string> functions;
    std::vector<std::string> events;
    std::map<std::string, std::string> colormap;
};

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.
using namespace libconfig;
using namespace std;
int config(char* config_file, Filter& filter)
{
    Config cfg;

    // Read the file. If there is an error, report it and exit.
    try {
        cfg.readFile(config_file);
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
        std::string name = cfg.lookup("name");
        std::cout << "Topic: " << name << std::endl
                  << std::endl;
    } catch (const SettingNotFoundException& nfex) {
        cerr << "No 'name' setting in configuration file." << std::endl;
    }

    // Get the mode.
    try {
        std::string tracing_mode = cfg.lookup("tracing_mode");
        filter.tracing_mode = tracing_mode;
        std::cout << "Tracing Mode: " << filter.tracing_mode << std::endl
                  << std::endl;
    } catch (const SettingNotFoundException& nfex) {
        std::cerr << "No 'mode' setting in configuration file." << std::endl;
    }

    const Setting& root = cfg.getRoot();

    // Output a list of all books in the inventory.
    try {
        const Setting& functions = root["symbols"]["functions"];
        int count = functions.getLength();

        std::cout << setw(40) << left << "Traced Symbols"
                  << "  "
                  << setw(40) << left << "Color"
                  << "   "
                  << std::endl;

        for (int i = 0; i < count; ++i) {
            const Setting& function = functions[i];
            // Only output the record if all of the expected fields are present.
            string name;
            string color;
            if (!(function.lookupValue("name", name)
                    && function.lookupValue("color", color))) {
                continue;
            }
            filter.functions.push_back(name);
            filter.colormap[name] = color;
            std::cout << setw(40) << left << name << "  "
                      << setw(40) << left << color << "  "
                      << std::endl;
        }
        std::cout << std::endl;
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

void dump_json(auto* pFileReport, auto& vec_ltr, auto& kf_map, auto filter, auto offset)
{
    uint64_t count = 0, downsample = 1;
    fprintf(pFileReport, "trace_data = [");

    if (filter.tracing_mode == "full") {
        fprintf(pFileReport, "\n{");

        fprintf(pFileReport, "name: 'All',");

        fprintf(pFileReport, "color: 'grey',");

        fprintf(pFileReport, "turboThreshold: %u, ", vec_ltr.size());

        fprintf(pFileReport, "data: [\n");
        for (auto trace : vec_ltr) {
            if ((count++) % downsample == 0) {
                int id_offset = 0;
                std::string tracename(trace.func_name);
                id_offset = offset;
                fprintf(pFileReport, "{ x: %lf, y: %d, name: \"%s\"},\n", trace.timestamp, kf_map[tracename] + id_offset, trace.func_name);
            }
        }
        fprintf(pFileReport, "]},\n");
    } else {
        for (auto keyword : filter.functions) {
            fprintf(pFileReport, "\n{");

            fprintf(pFileReport, "name: '%s',", keyword.c_str());

            fprintf(pFileReport, "color: '%s',", filter.colormap[keyword].c_str());

            fprintf(pFileReport, "turboThreshold: %u, ", vec_ltr.size());

            fprintf(pFileReport, "data: [\n");
            for (auto trace : vec_ltr) {
                if ((count++) % downsample == 0) {
                    int id_offset = 0;
                    std::string tracename(trace.func_name);
                    if (tracename.find(keyword) != std::string::npos) {
                        id_offset = offset;
                        fprintf(pFileReport, "{ x: %lf, y: %d, name: \"%s\"},\n", trace.timestamp, kf_map[tracename] + id_offset, trace.func_name);
                    }
                }
            }
            fprintf(pFileReport, "]},\n");
        }
    }
    fprintf(pFileReport, "]\n");
}


void dump_csv(auto* pFileReport, auto& vec_ltr, auto& kf_map, auto filter, auto offset)
{
    uint64_t count = 0, downsample = 1;
    bool bTraceAll = false;
    if (filter.tracing_mode == "full") {
        bTraceAll = true;
    }

    fprintf(pFileReport, "timestamp,func_id,func_name,color\n");

    for (std::vector<TraceRecord>::iterator it = vec_ltr.begin(); it != vec_ltr.end(); it++) {
        if ((count++) % downsample == 0) {
            int id_offset = 0;
            std::string tracename((*it).func_name);
            if (bTraceAll) {
                fprintf(pFileReport, "%lf,%d,%s,%s\n", (*it).timestamp,
                    kf_map[tracename],
                    tracename.c_str(),
                    "grey");
            } else {
                for (auto keyword : filter.functions) {
                    //std::cout<<"Filtered function = "
                    //          << keyword
                    //          << " with color "
                    //          << filter.colormap[keyword]
                    //          << std::endl;
                    if (tracename.find(keyword) != std::string::npos) {
                        id_offset = offset;
                        fprintf(pFileReport, "%lf,%d,%s,%s\n", (*it).timestamp,
                            kf_map[tracename] + id_offset,
                            tracename.c_str(),
                            filter.colormap[keyword].c_str());
                        break;
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    FILE *pFile, *pFileReport;
    char mystring[6000];
    std::vector<TraceRecord> vec_ltr;
    Filter filter;

    int t = 0;
    if (argc < 3) {
        printf("Usage: ./fsa defaut.cfg perf.script\n");
        return -1;
    }

    config(argv[1], filter);

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
    for (std::vector<TraceRecord>::iterator it = vec_ltr.begin(); it != vec_ltr.end(); ++it) {
        std::string key((*it).func_name);
        kf_map[key] = 1;
    }

    int kf_id = 0;
    for (std::map<std::string, int>::iterator it = kf_map.begin(); it != kf_map.end(); ++it) {
        (*it).second = kf_id += 10;
    }

    pFileReport = fopen("report.csv", "w");
    dump_csv(pFileReport, vec_ltr, kf_map, filter, 0);
    fclose(pFileReport);

    pFileReport = fopen("report.js", "w");
    dump_json(pFileReport, vec_ltr, kf_map, filter, 0);
    fclose(pFileReport);

    return 0;
}
