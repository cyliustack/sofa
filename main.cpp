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
    std::vector<std::string> nodes;
    std::vector<std::string> functions;
    std::vector<std::string> events;
    std::map<std::string, std::string> colormap;
    std::map<std::string, std::string> colormap4node;
};

class TraceFile {
public:
    std::string path;
    std::string color;
    FILE* pFile;
};

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.
using namespace libconfig;
using namespace std;
int config(char* config_file, auto& vec_tracefile, auto& filter)
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
        const Setting& tracefiles = root["tracefiles"];
        int count = tracefiles.getLength();

        std::cout << setw(40) << left << "TraceFile Name"
                  << "  "
                  << setw(40) << left << "Color"
                  << "   "
                  << std::endl;

        for (int i = 0; i < count; ++i) {
            const Setting& tracefile = tracefiles[i];
            // Only output the record if all of the expected fields are present.
            TraceFile tcfile;
            if (!(tracefile.lookupValue("path", tcfile.path)
                    && tracefile.lookupValue("color", tcfile.color))) {
                continue;
            }

            //tcfile.path  = path;
            //tcfile.color = color;
            char str_tmp[100] = { 0 };
            sprintf(str_tmp, "node%d", i);
            std::string node(str_tmp);
            filter.colormap4node[node] = tcfile.color;
            filter.nodes.push_back(node);

            vec_tracefile.push_back(tcfile);
            std::cout << setw(40) << left << tcfile.path << "  "
                      << setw(40) << left << tcfile.color << "  "
                      << std::endl;
        }
        std::cout << std::endl;
    } catch (const SettingNotFoundException& nfex) {
        // Ignore.
    }

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
    std::string node;
    double timestamp;
    uint64_t cycles;
    uint64_t addr;
    char proc_name[1000];
    char func_name[1000];
    void dump();
};

void TraceRecord::dump()
{
    printf("proc_name:%s, pid:%d, timestamp:%lf, cycles:%llu addr:%llu, function:%s node:%s\n", proc_name, pid, timestamp, cycles, addr, func_name, node.c_str());
}

void dump_json(auto* pFileReport, const auto& vec_ltr, auto& kf_map, auto& filter, auto offset)
{
    uint64_t count = 0, downsample = 1;
    fprintf(pFileReport, "trace_data = [");

    if (filter.tracing_mode == "full") {

        for (auto& node : filter.nodes) {
            fprintf(pFileReport, "\n{");

            fprintf(pFileReport, "name: '%s',", node.c_str());

            fprintf(pFileReport, "color: '%s',", filter.colormap4node[node].c_str());

            fprintf(pFileReport, "turboThreshold: %u, ", vec_ltr.size());

            fprintf(pFileReport, "data: [\n");
            for (auto& trace : vec_ltr) {
                if ((count++) % downsample == 0) {
                    int id_offset = 0;
                    std::string tracename(trace.func_name);
                    id_offset = offset;
                    if (trace.node == node) {
                        fprintf(pFileReport, "{ x: %lf, y: %d, name: \"%s\"},\n", trace.timestamp, kf_map[tracename] + id_offset, trace.func_name);
                    }
                }
            }
            fprintf(pFileReport, "]},\n");
        }

        //fprintf(pFileReport, "\n{");

        //fprintf(pFileReport, "name: 'All',");

        //fprintf(pFileReport, "color: 'grey',");

        //fprintf(pFileReport, "turboThreshold: %u, ", vec_ltr.size());

        //fprintf(pFileReport, "data: [\n");
        //for (auto& trace : vec_ltr) {
        //    if ((count++) % downsample == 0) {
        //        int id_offset = 0;
        //        std::string tracename(trace.func_name);
        //        id_offset = offset;
        //        fprintf(pFileReport, "{ x: %lf, y: %d, name: \"%s\"},\n", trace.timestamp, kf_map[tracename] + id_offset, trace.func_name);
        //    }
        //}
        //fprintf(pFileReport, "]},\n");
    } else {
        for (auto& keyword : filter.functions) {
            fprintf(pFileReport, "\n{");

            fprintf(pFileReport, "name: '%s',", keyword.c_str());

            fprintf(pFileReport, "color: '%s',", filter.colormap[keyword].c_str());

            fprintf(pFileReport, "turboThreshold: %u, ", vec_ltr.size());

            fprintf(pFileReport, "data: [\n");
            for (auto& trace : vec_ltr) {
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

    for (auto& trace : vec_ltr) {
        if ((count++) % downsample == 0) {
            int id_offset = 0;
            std::string tracename(trace.func_name);
            if (bTraceAll) {
                fprintf(pFileReport, "%lf,%d,%s,%s\n", trace.timestamp,
                    kf_map[tracename],
                    tracename.c_str(),
                    "grey");
            } else {
                for (auto& keyword : filter.functions) {
                    //std::cout<<"Filtered function = "
                    //          << keyword
                    //          << " with color "
                    //          << filter.colormap[keyword]
                    //          << std::endl;
                    if (tracename.find(keyword) != std::string::npos) {
                        id_offset = offset;
                        fprintf(pFileReport, "%lf,%d,%s,%s\n", trace.timestamp,
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
    std::vector<TraceFile> vec_tracefile;

    int t = 0;
    if (argc < 2) {
        printf("Usage: ./fsa defaut.cfg\n");
        return -1;
    }

    config(argv[1], vec_tracefile, filter);

    int nid = 0;
    for (auto& tracefile : vec_tracefile) {
        pFile = fopen(tracefile.path.c_str(), "r");
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
                sprintf(str_tmp, "node%d", nid);
                std::string node(str_tmp);
                tr.node = node;
                tr.dump();
                vec_ltr.push_back(tr);
            }
            nid++;
            fclose(pFile);
        }
    }

    std::map<std::string, int> kf_map;
    for (const auto& trace : vec_ltr) {
        std::string key(trace.func_name);
        kf_map[key] = 1;
    }

    int kf_id = 0;
    //    for (std::map<std::string, int>::iterator it = kf_map.begin(); it != kf_map.end(); ++it) {
    //        (*it).second = kf_id += 10;
    //    }

    for (auto& kf : kf_map) {
        kf.second = kf_id += 10;
    }

    pFileReport = fopen("report.csv", "w");
    dump_csv(pFileReport, vec_ltr, kf_map, filter, 0);
    fclose(pFileReport);

    pFileReport = fopen("report.js", "w");
    dump_json(pFileReport, vec_ltr, kf_map, filter, 0);
    fclose(pFileReport);

    return 0;
}
