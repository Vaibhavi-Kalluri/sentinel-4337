// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {Create2} from "@openzeppelin/contracts/utils/Create2.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {IEntryPoint} from "account-abstraction/interfaces/IEntryPoint.sol";
import {ISenderCreator} from "account-abstraction/interfaces/ISenderCreator.sol";
import {SentinelAccount} from "./SentinelAccount.sol";
import {SentinelPolicy} from "./SentinelPolicy.sol";

contract SentinelAccountFactory {
    SentinelAccount public immutable accountImplementation;
    ISenderCreator public immutable senderCreator;

    error NotSenderCreator(address sender, address expected);

    constructor(IEntryPoint entryPoint, SentinelPolicy policy) {
        accountImplementation = new SentinelAccount(entryPoint, policy);
        senderCreator = entryPoint.senderCreator();
    }

    function createAccount(address owner, uint256 salt)
        public
        returns (SentinelAccount account)
    {
        if (msg.sender != address(senderCreator)) {
            revert NotSenderCreator(msg.sender, address(senderCreator));
        }

        address accountAddress = getAddress(owner, salt);

        if (accountAddress.code.length > 0) {
            return SentinelAccount(payable(accountAddress));
        }

        account = SentinelAccount(
            payable(
                new ERC1967Proxy{salt: bytes32(salt)}(
                    address(accountImplementation),
                    abi.encodeWithSignature("initialize(address)", owner)
                )
            )
        );
    }

    function getAddress(address owner, uint256 salt)
        public
        view
        returns (address)
    {
        bytes32 bytecodeHash = keccak256(
            abi.encodePacked(
                type(ERC1967Proxy).creationCode,
                abi.encode(
                    address(accountImplementation),
                    abi.encodeWithSignature("initialize(address)", owner)
                )
            )
        );

        return Create2.computeAddress(bytes32(salt), bytecodeHash);
    }
}
