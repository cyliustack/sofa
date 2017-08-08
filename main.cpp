/***
Copyright (c) Jul. 2017, Cheng-Yueh Liu (cyliustack@gmail.com)
***/
#include "ppacket.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iostream>
#include <libconfig.h++>
#include <map>
#include <pcap.h>
#include <regex>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <string>
#include <typeinfo>
#include <unistd.h>
#include <vector>
using namespace libconfig;
using namespace std;

class Filter {
public:
    std::string tracing_mode;
    std::vector<std::string> nodes;
    std::vector<std::string> keywords;
    std::vector<std::string> events;
    std::map<std::string, std::string> colormap;
    std::map<std::string, std::string> colormap4node;
    int downsample;
};

class TraceFile {
public:
    std::string path;
    std::string color;
    FILE* pFile;
};

class SofaTime {
public:
    double t_begin;
    double t_end;
    SofaTime()
    {
        t_begin = 0;
        t_end = 0;
    };

    SofaTime(const double t_begin_in, double t_end_in)
    {
        t_begin = t_begin_in;
        t_end = t_end_in;
    };
};

int config(char* config_file, auto& tracefiles, auto& pcapfiles, auto& filter)
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
        const Setting& config_tracefiles = root["tracefiles"];
        int count = config_tracefiles.getLength();

        for (int i = 0; i < count; ++i) {
            const Setting& config_tracefile = config_tracefiles[i];
            // Only output the record if all of the expected fields are present.
            TraceFile tcfile;
            if (!(config_tracefile.lookupValue("path", tcfile.path)
                    && config_tracefile.lookupValue("color", tcfile.color))) {
                continue;
            }

            //tcfile.path  = path;
            //tcfile.color = color;
            char str_tmp[100] = { 0 };
            sprintf(str_tmp, "node%d", i);
            std::string node(str_tmp);
            filter.colormap4node[node] = tcfile.color;
            filter.nodes.push_back(node);

            tracefiles.push_back(tcfile);
        }
        std::cout << std::endl;
    } catch (const SettingNotFoundException& nfex) {
        // Ignore.
    }

    try {
        const Setting& keywords = root["symbols"]["keywords"];
        int count = keywords.getLength();

        std::cout << setw(40) << left << "Traced Symbols"
                  << "  "
                  << setw(40) << left << "Color"
                  << "   "
                  << std::endl;

        for (int i = 0; i < count; ++i) {
            const Setting& keyword = keywords[i];
            // Only output the record if all of the expected fields are present.
            string name;
            string color;
            if (!(keyword.lookupValue("name", name)
                    && keyword.lookupValue("color", color))) {
                continue;
            }
            filter.keywords.push_back(name);
            filter.colormap[name] = color;
            std::cout << setw(40) << left << name << "  "
                      << setw(40) << left << color << "  "
                      << std::endl;
        }
        std::cout << std::endl;
    } catch (const SettingNotFoundException& nfex) {
        // Ignore.
    }

    // Read tcpdump traces.
    try {
        const Setting& stn_pcapfiles = root["pcapfiles"];
        int count = stn_pcapfiles.getLength();

        //std::cout << setw(40) << left << "PcapFile Name"
        //          << "  "
        //          << setw(40) << left << "Color"
        //          << "   "
        //          << std::endl;

        for (int i = 0; i < count; ++i) {
            const Setting& stn_pcapfile = stn_pcapfiles[i];
            PcapFile pcapfile;
            if (!(stn_pcapfile.lookupValue("path", pcapfile.path)
                    && stn_pcapfile.lookupValue("color", pcapfile.color))) {
                continue;
            }

            //tcfile.path  = path;
            //tcfile.color = color;
            char str_tmp[100] = { 0 };
            pcapfiles.push_back(pcapfile);
            //std::cout << setw(40) << left << pcapfile.path << "  "
            //          << setw(40) << left << pcapfile.color << "  "
            //          << std::endl;
        }
        //std::cout << std::endl;
    } catch (const SettingNotFoundException& nfex) {
        // Ignore.
    }

    // Get the downsample.
    try {
        int downsample = cfg.lookup("downsample");
        filter.downsample = downsample;
        std::cout << "Downsample: " << filter.downsample << std::endl
                  << std::endl;
    } catch (const SettingNotFoundException& nfex) {
        std::cerr << "No 'downsample' setting in configuration file." << std::endl;
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

void dump_json(auto* pFileReport, const auto& traces, auto& kf_map, auto& filter, auto offset)
{
    uint64_t count = 0;
    fprintf(pFileReport, "trace_data = [");

    if (filter.tracing_mode == "full") {

        fprintf(pFileReport, "\n{");

        fprintf(pFileReport, "name: '%s',", "others");

        fprintf(pFileReport, "color: '%s',", "grey");

        fprintf(pFileReport, "turboThreshold: %u, ", traces.size());

        fprintf(pFileReport, "data: [\n");
        for (auto& trace : traces) {
            if ((count++) % filter.downsample == 0) {
                int id_offset = 0;
                std::string tracename(trace.func_name);
                fprintf(pFileReport, "{ x: %lf, y: %d, name: \"%s\"},\n", trace.timestamp, kf_map[tracename] + id_offset, trace.func_name);
            }
        }
        fprintf(pFileReport, "]},\n");
    }

    for (auto& keyword : filter.keywords) {
        fprintf(pFileReport, "\n{");

        fprintf(pFileReport, "name: '%s',", keyword.c_str());

        fprintf(pFileReport, "color: '%s',", filter.colormap[keyword].c_str());

        fprintf(pFileReport, "turboThreshold: %u, ", traces.size());

        fprintf(pFileReport, "data: [\n");
        for (auto& trace : traces) {
            if ((count++) % filter.downsample == 0) {
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
    fprintf(pFileReport, "]\n");
}

void dump_csv(auto* pFileReport, auto& traces, auto& kf_map, auto filter, auto offset)
{
    uint64_t count = 0, downsample = 1;
    bool bTraceAll = false;
    if (filter.tracing_mode == "full") {
        bTraceAll = true;
    }

    fprintf(pFileReport, "timestamp,func_id,func_name,color\n");

    for (auto& trace : traces) {
        if ((count++) % downsample == 0) {
            int id_offset = 0;
            std::string tracename(trace.func_name);
            if (bTraceAll) {
                fprintf(pFileReport, "%lf,%d,%s,%s\n", trace.timestamp,
                    kf_map[tracename],
                    tracename.c_str(),
                    "grey");
            } else {
                for (auto& keyword : filter.keywords) {
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

    //Read from trace files and store records into traces
    std::vector<TraceFile> tracefiles;
    std::vector<TraceRecord> traces;
    Filter filter;
    SofaTime sofa_time;

    //Read from pcap files and store records into packets
    std::vector<PcapFile> pcapfiles;
    std::vector<Packet> packets;

    int t = 0;
    if (argc < 2) {
        printf("Usage: ./fsa defaut.cfg\n");
        return -1;
    }

    pFile = fopen("sofa_time.txt", "r");
    if (pFile == NULL) {
        perror("Error opening sofa_time.txt");
        return -1;
    } else {
        if (fgets(mystring, sizeof(mystring), pFile) != NULL) {
            sscanf(mystring, "%lf", &sofa_time.t_begin);
            fclose(pFile);
        } else {
            perror("Nothing in sofa_time.txt");
            return -1;
        }
    }

    config(argv[1], tracefiles, pcapfiles, filter);

    int nid = 0;
    bool bGetFirstRecord = false;
    double timestamp_perf_begin = 0;
    for (auto& tracefile : tracefiles) {
        pFile = fopen(tracefile.path.c_str(), "r");
        if (pFile == NULL) {
            perror("Error opening file");
        } else {
            while (fgets(mystring, sizeof(mystring), pFile) != NULL) {
                TraceRecord tr;
                char str_tmp[2000];
                uint64_t timestamp;
                sscanf(mystring, "%s %d %s %lf: %d %s %x %s %s\n",
                    tr.proc_name,
                    &tr.pid,
                    str_tmp,
                    &tr.timestamp,
                    &tr.cycles,
                    str_tmp,
                    &tr.addr,
                    tr.func_name,
                    str_tmp);
                if (!bGetFirstRecord) {
                    timestamp_perf_begin = tr.timestamp;
                    bGetFirstRecord = true;
                }
                tr.timestamp = (tr.timestamp - timestamp_perf_begin) + sofa_time.t_begin;
                //ltr_tmp.kf_name = boost::core::demangle( func_name );
                sprintf(str_tmp, "node%d", nid);
                std::string node_name(str_tmp);
                tr.node = node_name;
                traces.push_back(tr);
            }
            nid++;
            fclose(pFile);
        }
    }

    for (auto& pcapfile : pcapfiles) {
        if (pcapfile.parse_from_file()) {
            for (auto& packet : pcapfile.packets) {
                TraceRecord tr;
                tr.timestamp = packet.timestamp-0.5;
                //ltr_tmp.kf_name = boost::core::demangle( func_name );
                sprintf(tr.func_name, "%d:network_event", packet.payload);
                tr.node = "node1";
                traces.push_back(tr);
            }
        }
    }

    std::map<std::string, int> kf_map;
    for (const auto& trace : traces) {
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
    dump_csv(pFileReport, traces, kf_map, filter, 0);
    fclose(pFileReport);

    pFileReport = fopen("report.js", "w");
    dump_json(pFileReport, traces, kf_map, filter, 0);
    fclose(pFileReport);

    return 0;
}
