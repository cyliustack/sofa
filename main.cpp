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
#include <pcap.h>

using namespace libconfig;
using namespace std;

class Filter {
public:
    std::string tracing_mode;
    std::vector<std::string> nodes;
    std::vector<std::string> functions;
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

class PcapFile : public TraceFile {
};

int pcap_demo(auto& filename);

int config(char* config_file, auto& vec_tracefile, auto& vec_pcapfile, auto& filter)
{
    Config cfg;

    // Read the file. If there is an error, report it and exit.
    try
    {
        cfg.readFile(config_file);
    }
    catch (const FileIOException& fioex)
    {
        std::cerr << "I/O error while reading file." << std::endl;
        return (EXIT_FAILURE);
    }
    catch (const ParseException& pex)
    {
        std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                  << " - " << pex.getError() << std::endl;
        return (EXIT_FAILURE);
    }

    // Get the store name.
    try
    {
        std::string name = cfg.lookup("name");
        std::cout << "Topic: " << name << std::endl
                  << std::endl;
    }
    catch (const SettingNotFoundException& nfex)
    {
        cerr << "No 'name' setting in configuration file." << std::endl;
    }

    // Get the mode.
    try
    {
        std::string tracing_mode = cfg.lookup("tracing_mode");
        filter.tracing_mode = tracing_mode;
        std::cout << "Tracing Mode: " << filter.tracing_mode << std::endl
                  << std::endl;
    }
    catch (const SettingNotFoundException& nfex)
    {
        std::cerr << "No 'mode' setting in configuration file." << std::endl;
    }

    const Setting& root = cfg.getRoot();

    // Output a list of all books in the inventory.
    try
    {
        const Setting& tracefiles = root["tracefiles"];
        int count = tracefiles.getLength();

        std::cout << setw(40) << left << "TraceFile Name"
                  << "  "
                  << setw(40) << left << "Color"
                  << "   "
                  << std::endl;

        for (int i = 0; i < count; ++i) {
            const Setting& stn_tracefile = tracefiles[i];
            // Only output the record if all of the expected fields are present.
            TraceFile tcfile;
            if (!(stn_tracefile.lookupValue("path", tcfile.path)
                  && stn_tracefile.lookupValue("color", tcfile.color))) {
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
    }
    catch (const SettingNotFoundException& nfex)
    {
        // Ignore.
    }

    try
    {
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
    }
    catch (const SettingNotFoundException& nfex)
    {
        // Ignore.
    }

    // Read tcpdump traces.
    try
    {
        const Setting& stn_pcapfiles = root["pcapfiles"];
        int count = stn_pcapfiles.getLength();

        std::cout << setw(40) << left << "PcapFile Name"
                  << "  "
                  << setw(40) << left << "Color"
                  << "   "
                  << std::endl;

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
            vec_pcapfile.push_back(pcapfile);
            std::cout << setw(40) << left << pcapfile.path << "  "
                      << setw(40) << left << pcapfile.color << "  "
                      << std::endl;
        }
        std::cout << std::endl;
    }
    catch (const SettingNotFoundException& nfex)
    {
        // Ignore.
    }
    
     // Get the downsample.
    try
    {
        int downsample = cfg.lookup("downsample");
        filter.downsample = downsample;
        std::cout << "Downsample: " << filter.downsample << std::endl
                  << std::endl;
    }
    catch (const SettingNotFoundException& nfex)
    {
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

void dump_json(auto* pFileReport, const auto& vec_ltr, auto& kf_map, auto& filter, auto offset)
{
    uint64_t count = 0;
    fprintf(pFileReport, "trace_data = [");

    if (filter.tracing_mode == "full") {

        for (auto& node : filter.nodes) {
            fprintf(pFileReport, "\n{");

            fprintf(pFileReport, "name: '%s',", node.c_str());

            fprintf(pFileReport, "color: '%s',", filter.colormap4node[node].c_str());

            fprintf(pFileReport, "turboThreshold: %u, ", vec_ltr.size());

            fprintf(pFileReport, "data: [\n");
            for (auto& trace : vec_ltr) {
                if ((count++) % filter.downsample == 0) {
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
    FILE* pFile, *pFileReport;
    char mystring[6000];
    std::vector<TraceRecord> vec_ltr;
    Filter filter;
    std::vector<TraceFile> vec_tracefile;
    std::vector<PcapFile> vec_pcapfile;

    int t = 0;
    if (argc < 2) {
        printf("Usage: ./fsa defaut.cfg\n");
        return -1;
    }

    config(argv[1], vec_tracefile, vec_pcapfile, filter);

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

    for (auto& pcapfile : vec_pcapfile) {
        pFile = fopen(pcapfile.path.c_str(), "r");
        if (pFile == NULL) {
            perror("Error opening pcap file\n");
        } else {
            fclose(pFile);
            pcap_demo(pcapfile.path);
        }
    }

    pFileReport = fopen("report.csv", "w");
    dump_csv(pFileReport, vec_ltr, kf_map, filter, 0);
    fclose(pFileReport);

    pFileReport = fopen("report.js", "w");
    dump_json(pFileReport, vec_ltr, kf_map, filter, 0);
    fclose(pFileReport);

    return 0;
}

int pcap_demo(auto& filename)
{

    /*
    * Step 3 - Create an char array to hold the error.
    */

    // Note: errbuf in pcap_open functions is assumed to be able to hold at least PCAP_ERRBUF_SIZE chars
    //       PCAP_ERRBUF_SIZE is defined as 256.
    // http://www.winpcap.org/docs/docs_40_2/html/group__wpcap__def.html
    char errbuff[PCAP_ERRBUF_SIZE];

    /*
    * Step 4 - Open the file and store result in pointer to pcap_t
    */

    // Use pcap_open_offline
    // http://www.winpcap.org/docs/docs_41b5/html/group__wpcapfunc.html#g91078168a13de8848df2b7b83d1f5b69
    pcap_t* pcap = pcap_open_offline(filename.c_str(), errbuff);

    /*
    * Step 5 - Create a header and a data object
    */

    // Create a header object:
    // http://www.winpcap.org/docs/docs_40_2/html/structpcap__pkthdr.html
    struct pcap_pkthdr* header;

    // Create a character array using a u_char
    // u_char is defined here:
    // C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\Include\WinSock2.h
    // typedef unsigned char   u_char;
    const u_char* data;

    /*
    * Step 6 - Loop through packets and print them to screen
    */
    u_int packetCount = 0;
    while (int returnValue = pcap_next_ex(pcap, &header, &data) >= 0) {
        // Print using printf. See printf reference:
        // http://www.cplusplus.com/reference/clibrary/cstdio/printf/

        // Show the packet number
        printf("Packet # %i\n", ++packetCount);

        // Show the size in bytes of the packet
        printf("Packet size: %d bytes\n", header->len);

        // Show a warning if the length captured is different
        if (header->len != header->caplen)
            printf("Warning! Capture size different than packet size: %ld bytes\n", header->len);

        // Show Epoch Time
        printf("Epoch Time: %d.%d seconds\n", header->ts.tv_sec, header->ts.tv_usec);

        // loop through the packet and print it as hexidecimal representations of octets
        // We also have a function that does this similarly below: PrintData()
        for (u_int i = 0; (i < header->caplen); i++) {
            // Start printing on the next after every 16 octets
            if ((i % 16) == 0)
                printf("\n");

            // Print each octet as hex (x), make sure there is always two characters (.2).
            printf("%.2x ", data[i]);
        }

        // Add two lines between packets
        printf("\n\n");
    }
}
