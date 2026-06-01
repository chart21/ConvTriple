/*
Copyright (c) 2018 Xiao Wang (wangxiao@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Enquiries about further applications and development opportunities are welcome.
*/

#ifndef NETWORK_IO_CHANNEL
#define NETWORK_IO_CHANNEL

#include "io_channel.hpp"
#include <iostream>
#include <memory> // std::align
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
using std::string;

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

enum class LastCall { None, Send, Recv };

namespace IO {

// constexpr size_t NETWORK_BUFFER_SIZE = 1024 * 16;
constexpr size_t NETWORK_BUFFER_SIZE = 1 << 20;

class NetIO : public IOChannel<NetIO> {
  public:
    bool is_server;
    int mysocket  = -1;
    int consocket = -1;
    FILE* stream  = nullptr;
    char* buffer  = nullptr;
    bool has_sent = false;
    string addr;
    int port;
    uint64_t num_rounds = 0;
    LastCall last_call  = LastCall::None;

    NetIO(NetIO&& other) noexcept {
        is_server    = other.is_server;
        mysocket     = other.mysocket;
        consocket    = other.consocket;
        stream       = other.stream;
        buffer       = other.buffer;
        has_sent     = other.has_sent;
        addr         = other.addr;
        port         = other.port;
        num_rounds   = other.num_rounds;
        last_call    = other.last_call;
        counter      = other.counter;
        recv_counter = other.recv_counter;

        other.mysocket  = -1;
        other.consocket = -1;
        other.stream    = nullptr;
        other.buffer    = nullptr;
    }

    NetIO(const NetIO& other) = delete;
    NetIO(const char* address, int port, bool quiet = false) {
        init_connection(address, port);
        if (quiet)
            std::cout << "connected\n";
    }

    void init_connection(const char* address, int port) {
        if (connected)
            return;
        this->port = port;
        is_server  = (address == nullptr);
        if (address == nullptr) {
            struct sockaddr_in dest;
            struct sockaddr_in serv;
            socklen_t socksize = sizeof(struct sockaddr_in);
            memset(&serv, 0, sizeof(serv));
            serv.sin_family      = AF_INET;
            serv.sin_addr.s_addr = htonl(INADDR_ANY); /* set our address to any interface */
            serv.sin_port        = htons(port);       /* set the server port number */
            mysocket             = socket(AF_INET, SOCK_STREAM, 0);
            int reuse            = 1;
            setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
            if (::bind(mysocket, (struct sockaddr*)&serv, sizeof(struct sockaddr)) < 0) {
                perror("error: bind");
                exit(1);
            }
            if (listen(mysocket, 1) < 0) {
                perror("error: listen");
                exit(1);
            }
            consocket = accept(mysocket, (struct sockaddr*)&dest, &socksize);
            close(mysocket);
        } else {
            addr = string(address);

            struct sockaddr_in dest;
            memset(&dest, 0, sizeof(dest));
            dest.sin_family      = AF_INET;
            dest.sin_addr.s_addr = inet_addr(address);
            dest.sin_port        = htons(port);

            while (1) {
                consocket = socket(AF_INET, SOCK_STREAM, 0);

                if (connect(consocket, (struct sockaddr*)&dest, sizeof(struct sockaddr)) == 0) {
                    break;
                }

                close(consocket);
                usleep(1000);
            }
        }
        char* wan_opt = getenv("CHEETAH_WAN_OPT");
        if (wan_opt && std::string(wan_opt) == "1") {
            set_nodelay();
        }
        stream = fdopen(consocket, "wb+");
        buffer = new char[NETWORK_BUFFER_SIZE];
        memset(buffer, 0, NETWORK_BUFFER_SIZE);
        // NOTE(Zhicong): we need _IONBF for the best network performance
        setvbuf(stream, buffer, _IOFBF, NETWORK_BUFFER_SIZE);
        connected = true;
    }

    void disconnect() {
        if (stream) {
            fflush(stream);
            fclose(stream);
            stream = nullptr;
        }
        if (consocket != -1) {
            close(consocket);
            consocket = -1;
        }
        if (buffer) {
            delete[] buffer;
            buffer = nullptr;
        }
        connected = false;
    }

    void sync() {
        int tmp = 0;
        if (is_server) {
            send_data_internal(&tmp, 1);
            flush();
            recv_data_internal(&tmp, 1);
        } else {
            recv_data_internal(&tmp, 1);
            send_data_internal(&tmp, 1);
            flush();
        }
    }

    ~NetIO() {
        if (stream) {
            fflush(stream);
            fclose(stream);
        }
        if (consocket != -1)
            close(consocket);
        if (buffer)
            delete[] buffer;
    }

    NetIO& operator=(const NetIO& other) = delete;

    NetIO& operator=(NetIO&& other) noexcept {
        if (&other == this)
            return *this;

        if (stream) {
            fflush(stream);
            fclose(stream);
        }
        if (consocket != -1)
            close(consocket);
        if (buffer)
            delete[] buffer;

        is_server    = other.is_server;
        mysocket     = other.mysocket;
        consocket    = other.consocket;
        stream       = other.stream;
        buffer       = other.buffer;
        has_sent     = other.has_sent;
        addr         = other.addr;
        port         = other.port;
        num_rounds   = other.num_rounds;
        last_call    = other.last_call;
        counter      = other.counter;
        recv_counter = other.recv_counter;

        other.mysocket  = -1;
        other.consocket = -1;
        other.stream    = nullptr;
        other.buffer    = nullptr;
        return *this;
    }

    void set_nodelay() {
        const int one = 1;
        setsockopt(consocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }

    void set_delay() {
        const int zero = 0;
        setsockopt(consocket, IPPROTO_TCP, TCP_NODELAY, &zero, sizeof(zero));
    }

    void flush() { fflush(stream); }

    void send_data_internal(const void* data, int len) {
        if (last_call != LastCall::Send) {
            num_rounds++;
            last_call = LastCall::Send;
        }
        int sent = 0;
        while (sent < len) {
            int res = fwrite(sent + (char*)data, 1, len - sent, stream);
            if (res >= 0)
                sent += res;
            else
                fprintf(stderr, "error: net_send_data %d\n", res);
        }
        has_sent = true;
    }

    void recv_data_internal(void* data, int len) {
        if (last_call != LastCall::Recv) {
            num_rounds++;
            last_call = LastCall::Recv;
        }
        if (has_sent)
            fflush(stream);
        has_sent = false;
        int sent = 0;
        while (sent < len) {
            int res = fread(sent + (char*)data, 1, len - sent, stream);
            if (res >= 0)
                sent += res;
            else
                fprintf(stderr, "error: net_send_data %d\n", res);
        }
    }

  private:
    bool connected = false;
};
/**@}*/

} // namespace IO
#endif // NETWORK_IO_CHANNEL
